from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from ..domain.config import RetrievalConfig
from ..domain.schemas import QueryAnalysis, RetrievedChunk
from ..infrastructure.index.parent_store import ParentRecord, ParentStore
from .parent_expansion_candidates import (
    allow_candidate,
    caption_candidate_matches_type,
    chunk_index,
    has_caption_like_signal,
    page_candidate_matches_type,
    rank_children_for_seed,
    same_page,
)
from .parent_expansion_debug import (
    build_parent_expansion_debug,
    initialize_optional_debug_reasons,
    record_parent_expansion_skip,
)
from .parent_expansion_plan import (
    build_caption_plan,
    build_non_caption_window_plan,
    comparison_caption_allowed,
    effective_limits,
    explicit_caption_query_type,
    false_table_trigger_guarded,
    is_caption_seed,
    is_table_hint_or_parameter_query,
    matching_caption_seed_docs,
    preferred_evidence_type,
    seed_caption_type,
    seed_matches_caption_query_type,
    select_mode,
    weak_caption_reference,
)


class ParentContextExpander:
    def __init__(self, parent_store: ParentStore | None, config: RetrievalConfig):
        self.parent_store = parent_store
        self.config = config

    def expand(
        self,
        question: str,
        seed_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
    ) -> tuple[list[RetrievedChunk], dict]:
        debug = build_parent_expansion_debug(self.config, seed_chunks, analysis)
        if not self.config.parent_expansion_enabled:
            debug["reason"] = "disabled"
            return list(seed_chunks), debug

        parent_index_path = Path(self.config.parent_index_path)
        if not parent_index_path.exists():
            debug["reason"] = "parent_index_missing"
            return list(seed_chunks), debug
        if self.parent_store is None:
            debug["reason"] = "parent_store_unavailable"
            return list(seed_chunks), debug
        if not seed_chunks:
            debug["reason"] = "no_seed_chunks"
            return [], debug

        caption_plan = self._build_caption_plan(question, seed_chunks)
        debug.update(
            {
                "figure_query": caption_plan["figure_query"],
                "table_query": caption_plan["table_query"],
                "caption_query_type": caption_plan["caption_query_type"],
                "caption_mode_trigger_source": caption_plan["caption_mode_trigger_source"],
                "false_table_trigger_guarded": caption_plan["false_table_trigger_guarded"],
                "caption_type_filter": caption_plan["caption_type_filter"],
                "caption_seed_docs": list(caption_plan["caption_seed_docs"]),
                "caption_target_doc_ids": list(caption_plan["caption_target_doc_ids"]),
                "target_doc_selection_reason": caption_plan["target_doc_selection_reason"],
            }
        )

        mode = self._select_mode(question, seed_chunks, analysis, caption_plan)
        local_context_plan = self._build_non_caption_window_plan(question, seed_chunks, mode, caption_plan)
        max_total, per_seed_limit, limit_reason = self._effective_limits(mode)
        debug["effective_intent"] = mode
        debug["effective_max_total"] = max_total
        debug["effective_per_seed_limit"] = per_seed_limit
        debug["limit_reason"] = limit_reason
        debug["primary_doc_window_gating"] = local_context_plan["enabled"]
        debug["window_target_doc_id"] = local_context_plan["target_doc_id"]
        debug["window_gating_reason"] = local_context_plan["reason"]
        debug["primary_doc_local_context_gating"] = local_context_plan["enabled"]
        debug["local_context_target_doc_id"] = local_context_plan["target_doc_id"]
        debug["local_context_gating_reason"] = local_context_plan["reason"]
        self._initialize_optional_debug_reasons(debug=debug, mode=mode, question=question, seed_chunks=seed_chunks)

        if max_total == 0:
            debug["reason"] = "max_total_zero"
            return [], debug

        final_chunks = list(seed_chunks[:max_total])
        seen_chunk_ids = {chunk.chunk_id for chunk in final_chunks}
        addable_total = max(0, max_total - len(final_chunks))
        if addable_total == 0:
            debug["reason"] = "seed_already_at_limit"
            debug["output_count"] = len(final_chunks)
            return final_chunks, debug

        per_seed_added: dict[str, int] = defaultdict(int)
        per_doc_added: dict[str, int] = defaultdict(int)
        added_chunk_ids: list[str] = []
        added_parent_ids: list[str] = []
        added_parent_types: list[str] = []

        if mode == "summary":
            summary_candidates = self._expand_summary(seed_chunks=seed_chunks, max_to_add=addable_total, debug=debug)
            for candidate, parent, reason, anchor_seed in summary_candidates:
                if candidate.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(candidate.chunk_id)
                final_chunks.append(self._clone_from_seed(candidate, anchor_seed, parent, reason))
                added_chunk_ids.append(candidate.chunk_id)
                added_parent_ids.append(parent.parent_id)
                added_parent_types.append(parent.parent_type)
                per_seed_added[anchor_seed.chunk_id] += 1
                per_doc_added[candidate.doc_id] += 1
                if len(added_chunk_ids) >= addable_total:
                    break
            debug["reason"] = "expanded" if added_chunk_ids else "no_candidates"
            debug["output_count"] = len(final_chunks)
            debug["added_chunk_ids"] = added_chunk_ids
            debug["added_parent_ids"] = added_parent_ids
            debug["added_parent_types"] = added_parent_types
            debug["per_seed_added"] = dict(per_seed_added)
            debug["per_doc_added"] = dict(per_doc_added)
            return final_chunks, debug

        seeds_to_expand = list(seed_chunks)
        if mode == "comparison":
            debug["comparison_mode"] = True
            debug["comparison_caption_allowed"] = self._comparison_caption_allowed(question)
            debug["comparison_seed_considered"] = [chunk.chunk_id for chunk in seeds_to_expand[:4]]
            debug["comparison_seed_skipped_by_rank"] = [chunk.chunk_id for chunk in seeds_to_expand[4:]]
            seeds_to_expand = seeds_to_expand[:4]

        for seed in seeds_to_expand:
            if len(added_chunk_ids) >= addable_total:
                break
            if per_seed_limit and per_seed_added[seed.chunk_id] >= per_seed_limit and mode != "summary":
                continue

            local_limit = addable_total - len(added_chunk_ids)
            if mode == "comparison":
                remaining_doc_budget = max(0, 1 - per_doc_added[seed.doc_id])
                if remaining_doc_budget <= 0:
                    debug["skipped_by_doc_cap"].append(seed.chunk_id)
                    continue
                local_limit = min(local_limit, remaining_doc_budget)
            elif mode != "summary" and per_seed_limit:
                local_limit = min(local_limit, per_seed_limit - per_seed_added[seed.chunk_id])
            if local_limit <= 0:
                continue

            candidates = self._collect_seed_candidates(
                seed=seed,
                question=question,
                mode=mode,
                max_to_add=local_limit,
                debug=debug,
            )
            for candidate, parent, reason in candidates:
                if candidate.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(candidate.chunk_id)
                final_chunks.append(self._clone_from_seed(candidate, seed, parent, reason))
                added_chunk_ids.append(candidate.chunk_id)
                added_parent_ids.append(parent.parent_id)
                added_parent_types.append(parent.parent_type)
                per_seed_added[seed.chunk_id] += 1
                per_doc_added[candidate.doc_id] += 1
                if len(added_chunk_ids) >= addable_total:
                    break
                if mode != "summary" and per_seed_limit and per_seed_added[seed.chunk_id] >= per_seed_limit:
                    break

        debug["reason"] = "expanded" if added_chunk_ids else "no_candidates"
        debug["output_count"] = len(final_chunks)
        debug["added_chunk_ids"] = added_chunk_ids
        debug["added_parent_ids"] = added_parent_ids
        debug["added_parent_types"] = added_parent_types
        debug["per_seed_added"] = dict(per_seed_added)
        debug["per_doc_added"] = dict(per_doc_added)
        debug["selected_parent_types"] = list(dict.fromkeys(added_parent_types))
        return final_chunks, debug

    def _collect_seed_candidates(
        self,
        *,
        seed: RetrievedChunk,
        question: str,
        mode: str,
        max_to_add: int,
        debug: dict,
    ) -> list[tuple[RetrievedChunk, ParentRecord, str]]:
        if self.parent_store is None or max_to_add <= 0:
            return []
        collected: list[tuple[RetrievedChunk, ParentRecord, str]] = []
        seen: set[str] = {seed.chunk_id}
        strategies = self._build_strategies(question=question, seed=seed, mode=mode, debug=debug)
        if mode == "caption":
            target_doc_ids = set(debug.get("caption_target_doc_ids") or [])
            if target_doc_ids and seed.doc_id not in target_doc_ids:
                debug["skipped_non_target_doc"].append(seed.chunk_id)
                return []

        evidence_used = False
        caption_used = False
        for parent_type, reason in strategies:
            parents = [parent for parent in self.parent_store.get_parents_for_chunk(seed.chunk_id) if parent.parent_type == parent_type]
            if parent_type == "page":
                debug["page_candidates_found"] += len(parents)
                if not parents and not debug["page_skipped_reason"]:
                    debug["page_skipped_reason"] = "no_parent_found"
            elif parent_type == "evidence_type_context":
                debug["evidence_candidates_found"] += len(parents)
                if not parents and not debug["evidence_skipped_reason"]:
                    debug["evidence_skipped_reason"] = "no_parent_found"
            elif parent_type == "caption_context":
                debug["caption_context_candidates_found"] += len(parents)
                debug["same_page_candidates_found"] += sum(
                    1
                    for parent in parents
                    for child in self._rank_children_for_seed(parent, seed)
                    if child.doc_id == seed.doc_id and self._same_page(seed, child)
                )
                debug["skipped_cross_doc"] += sum(
                    1
                    for parent in parents
                    for child in self._rank_children_for_seed(parent, seed)
                    if child.doc_id != seed.doc_id
                )

            if parent_type == "caption_context" and caption_used:
                debug["skipped_after_caption_limit"] += 1
                continue
            if parent_type == "page" and caption_used:
                if not debug["page_skipped_reason"]:
                    debug["page_skipped_reason"] = "lower_priority_parent_already_added"
                continue
            if (
                parent_type in {"chunk_window", "section_path"}
                and debug.get("primary_doc_local_context_gating")
                and seed.doc_id != debug.get("local_context_target_doc_id")
            ):
                debug["local_context_skipped_non_target_doc"].append(seed.chunk_id)
                debug["local_context_blocked_parent_types"].append(parent_type)
                if parent_type == "chunk_window":
                    debug["window_skipped_non_target_doc"].append(seed.chunk_id)
                if parent_type == "section_path":
                    debug["section_path_skipped_non_target_doc"].append(seed.chunk_id)
                continue

            parent_added = False
            for parent in parents:
                if parent_type == "caption_context" and parent.anchor_chunk_id and parent.anchor_chunk_id != seed.chunk_id:
                    continue
                if parent_type == "evidence_type_context":
                    if evidence_used:
                        continue
                    preferred = self._preferred_evidence_type(question, seed)
                    if preferred and parent.evidence_type != preferred:
                        continue
                ranked_children = self._rank_children_for_seed(parent, seed)
                if parent_type == "caption_context":
                    debug["caption_candidates_before_type_filter"] += len(ranked_children)
                elif parent_type == "page":
                    debug["page_candidates_before_filter"] += len(ranked_children)
                for child in ranked_children:
                    allowed, skip_reason = self._allow_candidate(
                        seed=seed,
                        candidate=child,
                        parent=parent,
                        question=question,
                        mode=mode,
                        parent_type=parent_type,
                        seen=seen,
                    )
                    if not allowed:
                        record_parent_expansion_skip(debug, parent_type, skip_reason)
                        continue
                    seen.add(child.chunk_id)
                    collected.append((child, parent, reason))
                    parent_added = True
                    if parent_type == "caption_context":
                        debug["caption_context_added"] += 1
                        debug["caption_candidates_added_by_type"] += 1
                        caption_used = True
                    elif parent_type == "page":
                        debug["page_context_added"] += 1
                        debug["page_candidates_added"] += 1
                        debug["page_fallback_used"] = True
                    elif parent_type == "evidence_type_context":
                        debug["evidence_candidates_added"] += 1
                        evidence_used = True
                    if len(collected) >= max_to_add:
                        return collected
                    break
            if parent_type == "page" and parents and not parent_added and not debug["page_skipped_reason"]:
                debug["page_skipped_reason"] = "lower_priority_parent_already_added" if caption_used else "duplicate_chunk"
            if parent_type == "evidence_type_context" and parents and not parent_added and not debug["evidence_skipped_reason"]:
                debug["evidence_skipped_reason"] = "lower_priority_parent_already_added" if collected else "duplicate_chunk"
        return collected

    def _expand_summary(
        self,
        *,
        seed_chunks: list[RetrievedChunk],
        max_to_add: int,
        debug: dict,
    ) -> list[tuple[RetrievedChunk, ParentRecord, str, RetrievedChunk]]:
        if self.parent_store is None or not self.config.parent_expansion_summary_sections_enabled:
            return []
        if not seed_chunks:
            return []
        target_sections = [section.lower() for section in self.config.parent_expansion_summary_sections]
        doc_counts = Counter(chunk.doc_id for chunk in seed_chunks if chunk.doc_id)
        top_docs = [doc_id for doc_id, _ in doc_counts.most_common()]
        debug["summary_docs_considered"] = top_docs
        existing_sections_by_doc: dict[str, set[str]] = defaultdict(set)
        anchor_seed_by_doc: dict[str, RetrievedChunk] = {}
        for chunk in seed_chunks:
            existing_sections_by_doc[chunk.doc_id].add((chunk.section or "").lower())
            anchor_seed_by_doc.setdefault(chunk.doc_id, chunk)

        candidates: list[tuple[RetrievedChunk, ParentRecord, str, RetrievedChunk]] = []
        seen = {chunk.chunk_id for chunk in seed_chunks}
        for doc_id in top_docs:
            anchor_seed = anchor_seed_by_doc.get(doc_id)
            if anchor_seed is None:
                continue
            for target_section in target_sections:
                if target_section in existing_sections_by_doc.get(doc_id, set()):
                    debug["summary_sections_skipped_existing"].append(f"{doc_id}:{target_section}")
                    continue
                found_for_target = False
                for parent in self.parent_store.get_parents_for_doc(doc_id, parent_type="section"):
                    if parent.section.lower() != target_section:
                        continue
                    for child in self._rank_children_for_seed(parent, anchor_seed):
                        if child.chunk_id in seen:
                            continue
                        seen.add(child.chunk_id)
                        candidates.append((child, parent, "summary_section", anchor_seed))
                        debug["summary_sections_added"].append(f"{doc_id}:{target_section}")
                        found_for_target = True
                        break
                    if len(candidates) >= max_to_add:
                        return candidates
                    if found_for_target:
                        break
                if found_for_target:
                    continue
                for parent in self.parent_store.get_parents_for_doc(doc_id, parent_type="section_path"):
                    if parent.section.lower() != target_section and parent.section_path_key.lower() != target_section:
                        continue
                    for child in self._rank_children_for_seed(parent, anchor_seed):
                        if child.chunk_id in seen:
                            continue
                        seen.add(child.chunk_id)
                        candidates.append((child, parent, "summary_section_path", anchor_seed))
                        debug["summary_sections_added"].append(f"{doc_id}:{target_section}")
                        found_for_target = True
                        break
                    if len(candidates) >= max_to_add:
                        return candidates
                    if found_for_target:
                        break
                if not found_for_target:
                    debug["summary_no_candidate_docs"].append(f"{doc_id}:{target_section}")
        return candidates

    def _build_strategies(
        self,
        *,
        question: str,
        seed: RetrievedChunk,
        mode: str,
        debug: dict,
    ) -> list[tuple[str, str]]:
        if mode == "comparison":
            if self._comparison_caption_allowed(question) and self._is_caption_seed(seed) and self.config.parent_expansion_caption_enabled:
                strategies: list[tuple[str, str]] = [("caption_context", "caption_context")]
                if self.config.parent_expansion_section_path_enabled:
                    strategies.append(("section_path", "section_path"))
            elif self.config.parent_expansion_section_path_enabled:
                strategies = [("section_path", "section_path")]
            else:
                strategies = []
            if self.config.parent_expansion_window_enabled:
                strategies.append(("chunk_window", "chunk_window"))
            return strategies

        strategies: list[tuple[str, str]] = []
        if mode == "caption":
            debug["caption_mode"] = True
            debug["caption_anchor_doc_id"] = seed.doc_id
            debug["same_doc_only"] = True
            if self._is_caption_seed(seed) and self.config.parent_expansion_caption_enabled:
                strategies.append(("caption_context", "caption_context"))
            if self.config.parent_expansion_page_enabled:
                strategies.append(("page", "page_context"))
            if self.config.parent_expansion_window_enabled:
                strategies.append(("chunk_window", "chunk_window"))
            return strategies

        if self.config.parent_expansion_window_enabled:
            strategies.append(("chunk_window", "chunk_window"))
        if self.config.parent_expansion_section_path_enabled:
            strategies.append(("section_path", "section_path"))
        if self.config.parent_expansion_evidence_enabled and self._preferred_evidence_type(question, seed):
            strategies.append(("evidence_type_context", "evidence_type_context"))
        return strategies

    def _select_mode(
        self,
        question: str,
        seed_chunks: list[RetrievedChunk],
        analysis: QueryAnalysis,
        caption_plan: dict | None = None,
    ) -> str:
        return select_mode(question, seed_chunks, analysis, caption_plan)

    def _effective_limits(self, mode: str) -> tuple[int, int, str]:
        return effective_limits(self.config, mode)

    def _rank_children_for_seed(self, parent: ParentRecord, seed: RetrievedChunk) -> list[RetrievedChunk]:
        return rank_children_for_seed(self.parent_store, parent, seed)

    def _clone_from_seed(
        self,
        candidate: RetrievedChunk,
        anchor: RetrievedChunk,
        parent: ParentRecord,
        reason: str,
    ) -> RetrievedChunk:
        cloned = RetrievedChunk(
            chunk_id=candidate.chunk_id,
            doc_id=candidate.doc_id,
            source_file=candidate.source_file,
            title=candidate.title,
            section=candidate.section,
            text=candidate.text,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            vector_score=anchor.vector_score,
            bm25_score=anchor.bm25_score,
            rerank_score=max((anchor.rerank_score or 0.0) - 0.01, 0.0),
            fusion_score=anchor.fusion_score,
            metadata=dict(candidate.metadata),
        )
        cloned.metadata.update(
            {
                "parent_expansion": True,
                "expanded_from_chunk_id": anchor.chunk_id,
                "expanded_from_parent_id": parent.parent_id,
                "expanded_from_parent_type": parent.parent_type,
                "parent_expansion_reason": reason,
            }
        )
        return cloned

    def _allow_candidate(
        self,
        *,
        seed: RetrievedChunk,
        candidate: RetrievedChunk,
        parent: ParentRecord,
        question: str,
        mode: str,
        parent_type: str,
        seen: set[str],
    ) -> tuple[bool, str]:
        return allow_candidate(
            seed=seed,
            candidate=candidate,
            parent=parent,
            question=question,
            mode=mode,
            parent_type=parent_type,
            seen=seen,
        )

    def _record_skip(self, debug: dict, parent_type: str, reason: str) -> None:
        record_parent_expansion_skip(debug, parent_type, reason)

    def _same_page(self, left: RetrievedChunk, right: RetrievedChunk) -> bool:
        return same_page(left, right)

    def _is_caption_seed(self, seed: RetrievedChunk) -> bool:
        return is_caption_seed(seed)

    def _initialize_optional_debug_reasons(
        self,
        *,
        debug: dict,
        mode: str,
        question: str,
        seed_chunks: list[RetrievedChunk],
    ) -> None:
        preferred = ""
        if seed_chunks:
            preferred = self._preferred_evidence_type(question, seed_chunks[0])
        initialize_optional_debug_reasons(
            debug=debug,
            config=self.config,
            mode=mode,
            seed_chunks=seed_chunks,
            preferred_evidence_type=preferred,
        )

    def _comparison_caption_allowed(self, question: str) -> bool:
        return comparison_caption_allowed(question)

    def _build_caption_plan(self, question: str, seed_chunks: list[RetrievedChunk]) -> dict:
        return build_caption_plan(question, seed_chunks)

    def _build_non_caption_window_plan(
        self,
        question: str,
        seed_chunks: list[RetrievedChunk],
        mode: str,
        caption_plan: dict,
    ) -> dict:
        return build_non_caption_window_plan(question, seed_chunks, mode, caption_plan)

    def _explicit_caption_query_type(self, question: str) -> str:
        return explicit_caption_query_type(question)

    def _weak_caption_reference(self, question: str) -> bool:
        return weak_caption_reference(question)

    def _false_table_trigger_guarded(self, question: str) -> bool:
        return false_table_trigger_guarded(question)

    def _matching_caption_seed_docs(self, seed_chunks: list[RetrievedChunk], query_type: str) -> list[str]:
        return matching_caption_seed_docs(seed_chunks, query_type)

    def _seed_matches_caption_query_type(self, seed: RetrievedChunk, query_type: str) -> bool:
        return seed_matches_caption_query_type(seed, query_type)

    def _seed_caption_type(self, seed: RetrievedChunk) -> str:
        return seed_caption_type(seed)

    def _caption_candidate_matches_type(self, candidate: RetrievedChunk, query_type: str) -> bool:
        return caption_candidate_matches_type(candidate, query_type)

    def _page_candidate_matches_type(self, candidate: RetrievedChunk, query_type: str) -> bool:
        return page_candidate_matches_type(candidate, query_type)

    def _has_caption_like_signal(self, candidate: RetrievedChunk) -> bool:
        return has_caption_like_signal(candidate)

    def _is_table_hint_or_parameter_query(self, question: str) -> bool:
        return is_table_hint_or_parameter_query(question)

    def _preferred_evidence_type(self, question: str, seed: RetrievedChunk) -> str:
        return preferred_evidence_type(question, seed)

def _chunk_index(chunk: RetrievedChunk) -> int:
    return chunk_index(chunk)
