#!/usr/bin/env python3
"""Phase 5F-3C strict main eval semantic enhancement.

The script only reads existing eval/chunk artifacts and writes reports under
reports/phase5f_eval_semantic_enhancement_v2. It does not run retrieval,
generation, RAGAS, OCR, index building, or project model services.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "phase5f_eval_semantic_enhancement_v2"

STRICT_MAIN_IN = ROOT / "reports" / "phase5f_eval_semantic_cleanup" / "strict_main_eval_set.jsonl"
LEXICAL_IN = ROOT / "reports" / "phase5f_eval_semantic_cleanup" / "lexical_stress_eval_set.jsonl"
DIAGNOSTIC_ACTIVE_IN = (
    ROOT / "reports" / "phase5f_eval_semantic_cleanup" / "diagnostic_active_eval_set.jsonl"
)
MAIN_AUDIT_IN = ROOT / "reports" / "phase5f_eval_semantic_cleanup" / "main_semantic_audit.csv"
REWRITE_LEDGER_IN = (
    ROOT / "reports" / "phase5f_eval_semantic_cleanup" / "query_rewrite_ledger.csv"
)
GOOD_NORMAL_IN = (
    ROOT / "reports" / "phase5f_normal_eval_quality_supplement" / "good_normal_control_merged.jsonl"
)
CLEAN_MAIN_IN = ROOT / "reports" / "phase5f_clean_eval_set" / "clean_main_eval_set.jsonl"
DIAGNOSTIC_IN = ROOT / "reports" / "phase5f_clean_eval_set" / "diagnostic_eval_set.jsonl"
QUALITY_LEDGER_IN = ROOT / "reports" / "phase5f_clean_eval_set" / "eval_quality_ledger.csv"
TARGET_MAPPING_AUDIT_IN = (
    ROOT / "reports" / "phase5f_clean_eval_set" / "target_mapping_audit.csv"
)

PREFERRED_CHUNK_PATHS = [
    Path("/tmp/biorag_phase5d3_caption_cleanup/chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase5c4_full_enhanced/chunks/chunks.jsonl"),
    Path("/tmp/biorag_phase4d_compact_chunks/chunks.jsonl"),
]

EXPECTED_INPUTS = [
    STRICT_MAIN_IN,
    LEXICAL_IN,
    DIAGNOSTIC_ACTIVE_IN,
    MAIN_AUDIT_IN,
    REWRITE_LEDGER_IN,
    GOOD_NORMAL_IN,
    CLEAN_MAIN_IN,
    DIAGNOSTIC_IN,
    QUALITY_LEDGER_IN,
    TARGET_MAPPING_AUDIT_IN,
    *PREFERRED_CHUNK_PATHS,
]

AUDIT_FIELDS = [
    "sample_id",
    "query_type",
    "original_query",
    "current_query",
    "target_doc_id",
    "stable_target_block_ids",
    "target_text_preview",
    "target_semantic_type",
    "target_semantic_summary",
    "user_intent",
    "answer_anchor",
    "can_form_natural_question",
    "natural_question_type",
    "hard_rule_violations",
    "current_hard_rule_violations",
    "final_hard_rule_violations",
    "semantic_decision",
    "rewritten_query",
    "rewrite_confidence",
    "final_label",
    "rationale",
]

REWRITE_FIELDS = [
    "sample_id",
    "query_type",
    "original_query",
    "current_query",
    "rewritten_query",
    "target_doc_id",
    "stable_target_block_ids",
    "target_semantic_summary",
    "user_intent",
    "rewrite_reason",
    "hard_rule_passed",
    "rewrite_confidence",
    "risk_if_kept",
]

NORMAL_RECOVERY_FIELDS = [
    "sample_id",
    "query",
    "rewritten_query",
    "query_was_rewritten",
    "target_doc_id",
    "stable_target_block_ids",
    "quality_label",
    "quality_risk_flags",
    "hard_rule_violations",
    "selected_for_main",
    "recovery_decision",
    "rationale",
]

DUP_FIELDS = [
    "check_type",
    "key",
    "count",
    "sample_ids",
    "query_types",
    "target_doc_ids",
    "notes",
]


NORMAL_QUERY_OVERRIDES = {
    "p5c5_normal_control_0014": "What did the GAS1 disruption experiment evaluate in Pichia pastoris?",
    "p5c5_normal_control_0015": "How did neuABC overexpression and nanK/nanA deletion improve sialylation efficiency?",
    "p5c5_normal_control_0019": "Which engineering changes redirected methanol flux toward malic acid production?",
    "p5c5_normal_control_0021": "How did the cofactor recycling system improve 3'-sialyllactose synthesis?",
    "p5c5_normal_control_0022": "How did orthogonal translation balance growth and biosynthesis in engineered bacteria?",
    "p5c5_normal_control_0023": "Which secretion-signal strategy improved protein secretion in Komagataella phaffii?",
    "p5c5_normal_control_0029": "Which secretory-pathway strategy improved hyaluronidase secretion in Pichia pastoris?",
    "phase5f2b_normal_supplement_0003": "Why was PNGase F treatment performed before MS/MS analysis?",
    "phase5f2b_normal_supplement_0006": "What measurement would indicate UPR or HSR induction in the study?",
    "phase5f2b_normal_supplement_0014": "What did SEM analysis investigate in laccase-treated sugarcane tops?",
    "phase5f2b_normal_supplement_0017": "What allergic responses were mitigated by administering the tested substances?",
    "phase5f2b_normal_supplement_0026": "What group differences were compared using one-way ANOVA?",
}

TABLE_CONTENT_SCOPE_REVIEW_QUERY_OVERRIDES = {
    "p5c2_table_content_0001": "What proteins were detected in P. pastoris culture media under the three fermentation conditions?",
    "p5c2_table_content_0024": "What E. coli strains and plasmids were used in the study?",
    "p5c3_table_content_0001": "What ethanol yields and glucose or xylose consumption rates were reported for engineered S. cerevisiae strains?",
    "p5c3_table_content_0003": "What ethanol yields and xylose or arabinose consumption rates were reported for engineered S. cerevisiae cultures?",
    "p5c3_table_content_0005": "What S. cerevisiae strain genotypes were listed for the engineering study?",
    "p5c3_table_content_0006": "What genes and proteins were upregulated in L. casei after bile exposure?",
    "p5c3_table_content_0007": "What E. coli strains and plasmids were used to construct the LNT-producing engineered strain?",
    "p5c3_table_content_0008": "What activity and selectivity parameters were compared for wild-type LnbB and its variants?",
    "p5c3_table_content_0009": "What leucine-lacking medium was used to further cultivate the yeast cells?",
    "p5c3_table_content_0010": "What association between HMO production and maternal GBS colonization was analyzed across milk groups?",
    "p5c3_table_content_0011": "What erythritol-production parameters were reported for screening and mutagenic strains?",
    "p5c3_table_content_0012": "What 2′FL concentration, absorption, and excretion measurements were reported across feeds, plasma, and urine?",
    "p5c3_table_content_0015": "What key characteristics were summarized for commonly used yeast promoters?",
    "p5c3_table_content_0018": "What constructed strains and plasmids were listed for the Bacillus subtilis study?",
    "p5c3_table_content_0020": "What recent advances were summarized for full-length antibody production in E. coli?",
    "p5c3_table_content_0022": "What strategies were summarized for overcoming recombinant protein expression problems in E. coli?",
    "p5c5_table_content_0001": "What S. cerevisiae strains were used for the optogenetic pyruvate decarboxylase study?",
    "p5c5_table_content_0009": "What E. coli strains and plasmids were used in the engineering study?",
    "p5c5_table_content_0012": "What peptidoglycan synthesis enzymes and cell morphogenesis proteins were listed for E. coli?",
    "p5c5_table_content_0023": "What in vivo LNT II and LNT production measurements were compared using E. coli?",
    "p5c5_table_content_0027": "What primer sequences were used for qRT-PCR assays?",
    "p5c5_table_content_0037": "What medium was used to culture E. coli in the N-acetylneuraminic acid bioproduction study?",
    "p5c5_table_content_0040": "What ROL activity and productivity measurements were compared across Pichia pastoris strains?",
    "p5c5_table_content_0066": "What bacterial strains and plasmids were used in the E. coli study?",
    "p5c5_table_content_0071": "What nutrient composition and content were reported for the infant milk powder samples?",
    "p5c5_table_content_0073": "How was HMO identity confirmed using NMR and mass spectrometry?",
    "p5c5_table_content_0079": "What annual production amounts and composition profiles were reported for agri-food wastes and by-products?",
    "p5c5_table_content_0081": "What primers were used in the Pichia pastoris NADH kinase overexpression study?",
    "p5c5_table_content_0083": "What primers were used in the Pichia pastoris rhamnose metabolic flux study?",
    "p5c5_table_content_0095": "What primers were listed for real-time PCR assays?",
    "p5c5_table_content_0097": "What fermentation conditions were used for the E. coli cultures?",
}

TABLE_CONTENT_SCOPE_REVIEW_DEMOTE_TO_LEXICAL = {
    "p5c5_table_content_0038": "What medium was used to culture E. coli in the N-acetylneuraminic acid bioproduction study?",
    "p5c5_table_content_0050": "Where were primers used for plasmid construction reported?",
    "p5c5_table_content_0063": "Where were primers used for plasmid construction reported?",
}

CAPTION_SCOPE_REVIEW_QUERY_OVERRIDES = {
    "p5c2_caption_level_0003": "What numbers of mothers and infants were colonised in each milk group?",
    "p5c2_caption_level_0006": "What P. pastoris strains were used in the study and what were they used for?",
    "p5c2_caption_level_0008": "What E. coli strains and plasmids were listed in the study?",
    "p5c2_caption_level_0010": "What kinetic parameters were assayed for the four SgCS wild types?",
    "p5c2_caption_level_0024": "What carbohydrate composition was reported for commercial Bimuno GOS powder and purified GOS-p solution?",
    "p5c2_caption_level_0026": "What fold-change effects of gene disruptions on HyHEL-Fab secretion were reported?",
    "p5c2_caption_level_0028": "What GEFIs were listed for measuring kinase activity?",
    "p5c3_caption_level_0037": "What P. pastoris strains were used for biotechnological characterization during bioreactor cultivations?",
    "p5c3_caption_level_0040": "What gene disruption targets were related to ERAD or proteasomal degradation?",
    "p5c5_figure_caption_0001": "What treatments were used to assess residual EntFuc hydrolysis activity?",
    "p5c5_figure_caption_0002": "What RNase B treatments were analyzed by MALDI-TOF MS?",
    "p5c5_figure_caption_0003": "What sequence alignment and docking comparisons were shown for Ev6ST and related sialyltransferases?",
    "p5c5_figure_caption_0004": "How was exogenous FAM20A localization assessed in 17IIA11 and LS8 cells?",
    "p5c5_figure_caption_0006": "What growth curves were measured for RHO1- and PKC1-expressing deletion strains?",
    "p5c5_figure_caption_0008": "What gene expression changes were measured in LS174T cells during the IL-13 challenge?",
    "p5c5_figure_caption_0009": "What strain comparison groups were used to identify growth-regulated genes?",
    "p5c5_figure_caption_0010": "How was phosphorylation of αs1-casein characterized in E. coli expression experiments?",
    "p5c5_figure_caption_0011": "What xanthohumol biosynthesis scheme and overexpression conditions were shown?",
    "p5c5_figure_caption_0013": "What fed-batch fermentation measurements were shown for NeuAc production by engineered strain BLNK-5?",
    "p5c5_figure_caption_0014": "What cytokine secretion measurements were shown after f-HMO treatment of Caco2 cells?",
    "p5c5_figure_caption_0016": "What ECM signaling pathway proteins were measured after 2′FL treatment?",
    "p5c5_figure_caption_0017": "What pathway transforms mannose-6-phosphate into GDP-L-fucose in microbial synthesis?",
    "p5c5_figure_caption_0026": "What measurements described how the H16_A3043/H16_A3044 system enhanced oil consumption and PHA production?",
    "p5c5_figure_caption_0028": "How was 2′-FL inhibition of Th17 cell infiltration and pro-inflammatory cytokines measured in the psoriasis model?",
    "p5c5_figure_caption_0033": "What fermentation products were measured during 24 h fermentation of HMOs and FOS/GOS controls?",
    "p5c5_figure_caption_0034": "How did lactate yields differ between neutral and sialylated HMOs during fermentation?",
    "p5c5_figure_caption_0035": "What microbial composition changes were shown after 24 h fermentation across HMO and control groups?",
    "p5c5_figure_caption_0044": "What CO2 mitigation pathway and PHA production route were illustrated?",
    "p5c5_figure_caption_0047": "What carbon-footprint scenario was mapped for improved PHA production?",
}

CAPTION_SCOPE_REVIEW_DEMOTE_TO_LEXICAL = {
    "p5c5_caption_level_table_0043": "Where were primers used for plasmid construction presented?",
}

METADATA_SCOPE_REVIEW_OVERRIDES = {
    "p5c3_table_content_0001": {
        "target_semantic_type": "production_or_titer_results",
        "target_semantic_summary": "ethanol yields and glucose or xylose consumption rates for engineered S. cerevisiae strains grown in lignocellulosic hydrolysates",
        "user_intent": "retrieve table-related text reporting ethanol yields and sugar consumption rates for engineered S. cerevisiae strains",
        "answer_anchor": "table caption or nearby table-related text",
    },
    "p5c3_table_content_0003": {
        "target_semantic_type": "production_or_titer_results",
        "target_semantic_summary": "ethanol yields and xylose or arabinose consumption rates for engineered S. cerevisiae cultures grown in synthetic media",
        "user_intent": "retrieve table-related text reporting ethanol yields and pentose consumption rates for engineered S. cerevisiae cultures",
        "answer_anchor": "table caption or nearby table-related text",
    },
    "p5c3_table_content_0015": {
        "target_semantic_type": "other",
        "target_semantic_summary": "commonly used yeast promoters and their key characteristics",
        "user_intent": "retrieve table-related text summarizing commonly used yeast promoters and key characteristics",
        "answer_anchor": "table caption or nearby table-related text",
    },
    "p5c5_table_content_0037": {
        "target_semantic_type": "fermentation_or_cultivation_conditions",
        "target_semantic_summary": "LB medium used to culture E. coli in the N-acetylneuraminic acid bioproduction study",
        "user_intent": "retrieve table-related text reporting the E. coli culture medium for the N-acetylneuraminic acid bioproduction study",
        "answer_anchor": "table caption or nearby table-related text",
    },
    "p5c5_table_content_0038": {
        "target_semantic_type": "fermentation_or_cultivation_conditions",
        "target_semantic_summary": "duplicate LB medium statement for E. coli culture in the N-acetylneuraminic acid bioproduction study",
        "user_intent": "treat duplicate E. coli culture-medium target as lexical stress rather than strict main",
        "answer_anchor": "table caption or nearby table-related text",
    },
    "p5c5_table_content_0050": {
        "target_semantic_type": "primers_oligonucleotides",
        "target_semantic_summary": "supplementary Table S1 pointer for primers used in plasmid construction",
        "user_intent": "treat supplementary primer-table pointer as lexical stress rather than strict main",
        "answer_anchor": "table caption or nearby table-related text",
    },
    "p5c5_table_content_0063": {
        "target_semantic_type": "primers_oligonucleotides",
        "target_semantic_summary": "supplementary Table S1 pointer for primers used in plasmid construction",
        "user_intent": "treat supplementary primer-table pointer as lexical stress rather than strict main",
        "answer_anchor": "table caption or nearby table-related text",
    },
    "p5c5_table_content_0081": {
        "target_semantic_type": "primers_oligonucleotides",
        "target_semantic_summary": "primers used in the Pichia pastoris NADH kinase overexpression study",
        "user_intent": "retrieve table-related text listing primers for the Pichia pastoris NADH kinase overexpression study",
        "answer_anchor": "table caption or nearby table-related text",
    },
    "p5c5_table_content_0083": {
        "target_semantic_type": "primers_oligonucleotides",
        "target_semantic_summary": "primers used in the Pichia pastoris rhamnose metabolic flux study",
        "user_intent": "retrieve table-related text listing primers for the Pichia pastoris rhamnose metabolic flux study",
        "answer_anchor": "table caption or nearby table-related text",
    },
    "p5c5_figure_caption_0016": {
        "target_semantic_type": "figure_experiment_result",
        "target_semantic_summary": "ECM signaling pathway proteins measured after 2′FL treatment",
        "user_intent": "retrieve the figure caption describing ECM signaling pathway proteins measured after 2′FL treatment",
        "answer_anchor": "figure caption text",
    },
    "p5c5_figure_caption_0017": {
        "target_semantic_type": "figure_experiment_result",
        "target_semantic_summary": "de novo GDP-L-fucose synthesis pathway from mannose-6-phosphate",
        "user_intent": "retrieve the figure caption describing the de novo GDP-L-fucose synthesis pathway",
        "answer_anchor": "figure caption text",
    },
    "p5c5_figure_caption_0044": {
        "target_semantic_type": "figure_experiment_result",
        "target_semantic_summary": "CO2 mitigation pathway and PHA production route",
        "user_intent": "retrieve the figure caption illustrating the CO2 mitigation pathway and PHA production route",
        "answer_anchor": "figure caption text",
    },
    "p5c5_figure_caption_0047": {
        "target_semantic_type": "figure_experiment_result",
        "target_semantic_summary": "carbon-footprint scenario for improved PHA production",
        "user_intent": "retrieve the figure caption mapping the carbon-footprint scenario for improved PHA production",
        "answer_anchor": "figure caption text",
    },
}

STRICT_MAIN_V2_SAMPLE_ID_ALLOWLIST = {
    "p5c2_table_content_0001",
    "p5c2_table_content_0024",
    "p5c3_table_content_0001",
    "p5c3_table_content_0003",
    "p5c3_table_content_0005",
    "p5c3_table_content_0006",
    "p5c3_table_content_0007",
    "p5c3_table_content_0008",
    "p5c3_table_content_0009",
    "p5c3_table_content_0010",
    "p5c3_table_content_0011",
    "p5c3_table_content_0012",
    "p5c3_table_content_0015",
    "p5c3_table_content_0018",
    "p5c3_table_content_0020",
    "p5c3_table_content_0022",
    "p5c5_table_content_0001",
    "p5c5_table_content_0009",
    "p5c5_table_content_0012",
    "p5c5_table_content_0023",
    "p5c5_table_content_0027",
    "p5c5_table_content_0037",
    "p5c5_table_content_0040",
    "p5c5_table_content_0066",
    "p5c5_table_content_0071",
    "p5c5_table_content_0073",
    "p5c5_table_content_0079",
    "p5c5_table_content_0081",
    "p5c5_table_content_0083",
    "p5c5_table_content_0095",
    "p5c5_table_content_0097",
    "p5c2_caption_level_0003",
    "p5c2_caption_level_0006",
    "p5c2_caption_level_0008",
    "p5c2_caption_level_0010",
    "p5c2_caption_level_0024",
    "p5c2_caption_level_0026",
    "p5c2_caption_level_0028",
    "p5c3_caption_level_0037",
    "p5c3_caption_level_0040",
    "p5c5_figure_caption_0001",
    "p5c5_figure_caption_0002",
    "p5c5_figure_caption_0003",
    "p5c5_figure_caption_0004",
    "p5c5_figure_caption_0006",
    "p5c5_figure_caption_0008",
    "p5c5_figure_caption_0009",
    "p5c5_figure_caption_0010",
    "p5c5_figure_caption_0011",
    "p5c5_figure_caption_0013",
    "p5c5_figure_caption_0014",
    "p5c5_figure_caption_0016",
    "p5c5_figure_caption_0017",
    "p5c5_figure_caption_0026",
    "p5c5_figure_caption_0028",
    "p5c5_figure_caption_0033",
    "p5c5_figure_caption_0034",
    "p5c5_figure_caption_0035",
    "p5c5_figure_caption_0044",
    "p5c5_figure_caption_0047",
    "p4e3_normal_supplement_0002",
    "p4e3_normal_supplement_0003",
    "p4e3_normal_supplement_0007",
    "p4e3_normal_supplement_0009",
    "p4e3_normal_supplement_0010",
    "p4e3_normal_supplement_0011",
    "p4e3_normal_supplement_0017",
    "p4e3_normal_supplement_0018",
    "p5c5_normal_control_0013",
    "p5c5_normal_control_0014",
    "p5c5_normal_control_0015",
    "p5c5_normal_control_0016",
    "p5c5_normal_control_0019",
    "p5c5_normal_control_0021",
    "p5c5_normal_control_0022",
    "p5c5_normal_control_0023",
    "p5c5_normal_control_0024",
    "p5c5_normal_control_0025",
    "p5c5_normal_control_0027",
    "p5c5_normal_control_0029",
    "phase5f2b_normal_supplement_0003",
    "phase5f2b_normal_supplement_0005",
    "phase5f2b_normal_supplement_0006",
    "phase5f2b_normal_supplement_0007",
    "phase5f2b_normal_supplement_0012",
    "phase5f2b_normal_supplement_0014",
    "phase5f2b_normal_supplement_0017",
    "phase5f2b_normal_supplement_0021",
    "phase5f2b_normal_supplement_0025",
    "phase5f2b_normal_supplement_0026",
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "used",
    "was",
    "were",
    "what",
    "where",
    "which",
    "with",
}

QUESTION_PREFIXES = {
    "which",
    "what",
    "where",
    "how",
    "why",
    "when",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
}

GENERIC_TECH_ALLOW = {"hmo", "hmos", "dna", "rna", "pcr", "qrt-pcr", "ms", "lc-ms", "maldi-tof"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def stable_ids(row: dict[str, Any]) -> list[str]:
    value = row.get("stable_target_block_ids") or []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        if "|" in value:
            return [v for v in value.split("|") if v]
        return [value] if value else []
    return []


def stable_ids_s(row: dict[str, Any]) -> str:
    return "|".join(stable_ids(row))


def row_target_key(row: dict[str, Any]) -> str:
    ids = stable_ids_s(row)
    return f"{row.get('target_doc_id', '')}|{ids or row.get('target_chunk_id_candidate', '')}"


def word_tokens(text: str) -> list[str]:
    return re.findall(
        r"[A-Za-z0-9]+(?:[.'/-][A-Za-z0-9]+)*|[\u4e00-\u9fff]+",
        normalize_ws(text),
    )


def english_tokens(text: str) -> list[str]:
    return [t for t in word_tokens(text) if re.search(r"[A-Za-z0-9]", t)]


def lower_word_tokens(text: str) -> list[str]:
    return [t.lower().strip(".,;:()[]{}") for t in english_tokens(text) if t.strip()]


def issue_join(issues: list[str]) -> str:
    return "; ".join(dict.fromkeys(i for i in issues if i))


def remove_caption_markers(text: str) -> str:
    cleaned = normalize_ws(text)
    cleaned = re.sub(r"^\[(?:TABLE|FIGURE)?\s*CAPTION\]\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:TABLE|FIGURE)\s+CAPTION\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bCAPTION\b", "", cleaned, flags=re.I)
    return normalize_ws(cleaned)


def remove_table_figure_prefix(text: str) -> str:
    cleaned = remove_caption_markers(text)
    cleaned = re.sub(
        r"^(?:Table|TABLE|Fig\.?|Figure)\s+[S]?[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)?\s*"
        r"[:.|-]?\s*",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"^\(?continued\)?\.?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^[A-Z]\.\s+", "", cleaned)
    cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return normalize_ws(cleaned)


def clean_preview_for_output(text: str) -> str:
    cleaned = remove_caption_markers(text)
    cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", cleaned)
    return normalize_ws(cleaned)


def technical_token(token: str) -> bool:
    token = token.strip(".,;:()[]{}")
    if not token:
        return False
    low = token.lower()
    if low in STOP_WORDS or low in GENERIC_TECH_ALLOW:
        return False
    if len(token) >= 28:
        return True
    if re.search(r"\d", token):
        return True
    if any(ch in token for ch in ["/", "_", "+"]):
        return True
    if "-" in token and not re.fullmatch(r"(?:co|non|high|low|full|whole|one|fed)-[a-z]+", low):
        return True
    letters = re.sub(r"[^A-Za-z]", "", token)
    if len(letters) >= 3:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if upper_ratio >= 0.45:
            return True
    return False


def numeric_count(tokens: list[str]) -> int:
    return sum(1 for token in tokens if re.search(r"\d", token))


def max_shared_ngram(query: str, target: str) -> int:
    q = lower_word_tokens(query)
    t = lower_word_tokens(target)
    if not q or not t:
        return 0
    max_len = 0
    positions: defaultdict[str, list[int]] = defaultdict(list)
    for idx, token in enumerate(t):
        positions[token].append(idx)
    for i, token in enumerate(q):
        for j in positions.get(token, []):
            length = 0
            while i + length < len(q) and j + length < len(t) and q[i + length] == t[j + length]:
                length += 1
            max_len = max(max_len, length)
    return max_len


def table_body(query: str) -> str:
    q = normalize_ws(query).rstrip("?")
    patterns = [
        r"^Which table (?:summarizes|summarises|reports|contains|lists|compares|describes|gives)\s+(.+)$",
        r"^Where can I find\s+(.+)$",
        r"^Where are\s+(.+?)\s+summarized in a table$",
    ]
    for pattern in patterns:
        match = re.match(pattern, q, flags=re.I)
        if match:
            return normalize_ws(match.group(1))
    return q


def hard_rule_lint(query: str, target_preview: str = "", query_type: str = "") -> list[str]:
    query = normalize_ws(query)
    target_clean = remove_caption_markers(target_preview)
    issues: list[str] = []
    if not query:
        return ["empty_query"]
    if re.search(r"\b(?:CAPTION|TABLE CAPTION|FIGURE CAPTION)\b", query, flags=re.I):
        issues.append("artifact_caption_marker")
    if re.search(r"\bDOCX?\b", query):
        issues.append("artifact_doc_marker")
    if re.search(r"Supplementary material artifact", query, flags=re.I):
        issues.append("artifact_supplementary_material")
    if re.search(r"\bTable\s+1\.\s*E\b", query, flags=re.I) or re.search(
        r"\bFig\.\s*3\.\s*$", query, flags=re.I
    ):
        issues.append("artifact_table_or_figure_stub")
    if re.search(r"\bcontinued\b", query, flags=re.I) and len(english_tokens(query)) <= 8:
        issues.append("artifact_continued_primary_content")
    if re.search(r"\b(?:con trol|pre sented|hydrol-\s*ysis|pro-\s*duction|fer-\s*mentation)\b", query, flags=re.I):
        issues.append("parser_linebreak_artifact")

    tokens = english_tokens(query)
    token_count = len(tokens)
    if re.search(r"[\u4e00-\u9fff]", query):
        if token_count <= 3 and len(query) < 12:
            issues.append("chinese_query_fragment")
    else:
        if token_count < 8:
            issues.append("query_too_short")
        if 25 <= token_count <= 28:
            issues.append("query_length_needs_strong_reason")
        if token_count > 28:
            issues.append("query_too_long")

    if max_shared_ngram(query, target_clean) > 7:
        issues.append("copies_target_preview_over_7_words")
    body = table_body(query)
    if re.match(r"^Which table summarizes\b", query, flags=re.I):
        body_tokens = english_tokens(body)
        if len(body_tokens) > 12 or max_shared_ngram(body, target_clean) > 6:
            issues.append("caption_compression_template")
    if re.match(r"^Which table (?:reports|contains|lists)\b", query, flags=re.I):
        body_tokens = english_tokens(body)
        tech_count = sum(1 for token in body_tokens if technical_token(token))
        if tech_count >= 4 or numeric_count(body_tokens) >= 3:
            issues.append("table_field_or_token_soup_template")
    if re.match(r"^Where are .+ summarized in a table\??$", query, flags=re.I):
        issues.append("mechanical_where_table_template")
    if re.match(r"^Which figure shows\b", query, flags=re.I):
        body_tokens = english_tokens(body)
        tech_count = sum(1 for token in body_tokens if technical_token(token))
        if tech_count > 4:
            issues.append("figure_caption_token_soup")
    if re.match(r"^What does (?:Table|Figure|Fig\.)\s+\S+\s+(?:report|show|describe)", query, flags=re.I):
        issues.append("generic_table_or_figure_question")

    tech_tokens = [token for token in tokens if technical_token(token)]
    nums = numeric_count(tokens)
    if len(tech_tokens) > 4:
        issues.append("too_many_id_or_gene_tokens")
    if nums > 3:
        issues.append("too_many_numeric_tokens")
    if re.search(r"\b[ACGTUNacgtun]{14,}\b", query):
        issues.append("sequence_or_primer_token")
    if token_count and len(tech_tokens) / max(token_count, 1) > 0.42 and len(tech_tokens) >= 3:
        issues.append("uppercase_or_id_density_high")

    grammar_patterns = [
        r"\bWhich table summarizes .* in E\??$",
        r"\bWhich table summarizes .* used in This Study\b",
        r"\bWhich table reports standard and low manufacturing cost factors were employed\b",
        r"\bWhich table summarizes all primers used .* are listed\b",
        r"\bWhich table summarizes the .*\b(?:were|was|are|is)\b",
        r"\bWhich table summarizes summarizes\b",
    ]
    if any(re.search(pattern, query, flags=re.I) for pattern in grammar_patterns):
        issues.append("grammar_incomplete_or_caption_syntax")
    if (
        query_type in {"table_content", "caption_level_table"}
        and re.search(
            r"\b(table\s+row|table\s+cell|row value|cell value|column value|specific row|specific cell)\b",
            query,
            flags=re.I,
        )
    ):
        issues.append("row_cell_structured_table_query")
    if re.search(r"\b(OCR|image-only|read the image|panel [A-Z]|gel band|micrograph)\b", query, flags=re.I):
        issues.append("ocr_or_image_query")
    return list(dict.fromkeys(issues))


def target_preview_looks_fragment(preview: str) -> bool:
    raw = normalize_ws(preview)
    cleaned = remove_table_figure_prefix(raw)
    words = [w for w in english_tokens(cleaned) if w.lower() not in {"continued", "cont"}]
    if re.search(r"\bcontinued\b", raw, flags=re.I) and len(words) < 10:
        return True
    if re.search(r"\b(?:Table|Fig\.?|Figure)\s+[S]?[A-Za-z0-9]+\.?\s*E\.?\s*$", raw, flags=re.I):
        return True
    if re.match(
        r"^\[(?:TABLE|FIGURE)?\s*CAPTION\]\s*(?:Table|Fig\.?|Figure)\s+[S]?[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)?\.?\s*$",
        raw,
        flags=re.I,
    ):
        return True
    return len(words) <= 3


def parser_artifact_present(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:con trol|pre sented|hydrol-\s*ysis|pro-\s*duction|fer-\s*mentation|high lighted|pin pointed|ana- lytical)\b",
            text,
            flags=re.I,
        )
    )


def load_chunk_texts(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    wanted = {str(row.get("target_chunk_id_candidate") or "") for row in rows}
    wanted.discard("")
    chunks: dict[str, dict[str, str]] = {}
    for path in PREFERRED_CHUNK_PATHS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk_id = str(chunk.get("chunk_id") or "")
                if chunk_id in wanted and chunk_id not in chunks:
                    chunks[chunk_id] = {
                        "text": normalize_ws(chunk.get("text", "")),
                        "title": normalize_ws(chunk.get("title", "")),
                    }
        if wanted.issubset(chunks):
            break
    return chunks


def choose_summary_source(row: dict[str, Any], chunks: dict[str, dict[str, str]]) -> str:
    preview = normalize_ws(row.get("target_text_preview", ""))
    if target_preview_looks_fragment(preview):
        chunk = chunks.get(str(row.get("target_chunk_id_candidate") or ""), {})
        text = normalize_ws(chunk.get("text", ""))
        if text:
            return text
    return preview


def target_summary_text(row: dict[str, Any], chunks: dict[str, dict[str, str]]) -> str:
    source = choose_summary_source(row, chunks)
    cleaned = remove_table_figure_prefix(source)
    cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 340:
        cleaned = cleaned[:340].rsplit(" ", 1)[0]
    return cleaned


def target_contains_field_list(summary: str) -> bool:
    patterns = [
        r"\b(strains?/plasmids?|strains? or plasmids?)\s+(?:characteristics|description|source)\b",
        r"\bstrain\s+genotype\s+source\b",
        r"\bplasmids?\s+description\s+source\b",
        r"\bprimer\s+(?:sequence|name)\b",
        r"\b(?:ratio|units|source|reference|description|characteristics)\b.*\b(?:ratio|units|source|reference|description|characteristics)\b",
        r"\b(?:C/N ratio|C/P ratio|Lignin/N ratio)\b",
    ]
    return any(re.search(pattern, summary, flags=re.I) for pattern in patterns)


def dense_ids_or_numbers(summary: str) -> bool:
    tokens = english_tokens(summary)
    if not tokens:
        return False
    tech = sum(1 for token in tokens if technical_token(token))
    nums = numeric_count(tokens)
    return tech >= 8 or nums >= 6 or tech / max(len(tokens), 1) >= 0.28


def organism_phrase(text: str) -> str:
    patterns = [
        (r"\bEscherichia coli\b|\bE\. coli\b", "E. coli"),
        (r"\bSaccharomyces cerevisiae\b|\bS\. cerevisiae\b|cerevisiae", "S. cerevisiae"),
        (r"\bPichia pastoris\b|\bP\. pastoris\b", "P. pastoris"),
        (r"\bKomagataella phaffii\b|\bK\. phaffii\b", "K. phaffii"),
        (r"\bBacillus subtilis\b|\bB\. subtilis\b", "B. subtilis"),
        (r"\bLactobacillus casei\b|\bL\. casei\b", "L. casei"),
        (r"\bBifidobacterium longum\b|\bB\. longum\b", "B. longum"),
        (r"\bLactococcus lactis\b|\bL\. lactis\b", "L. lactis"),
    ]
    for pattern, replacement in patterns:
        if re.search(pattern, text, flags=re.I):
            return replacement
    return ""


def product_phrase(text: str) -> str:
    candidates = [
        (r"\b3['-]?sialyllactose\b|\b3.?SL\b", "3'-sialyllactose"),
        (r"\b2['-]?fucosyllactose\b|\b2.?FL\b", "2'-fucosyllactose"),
        (r"\b3-FL\b|\b3['-]?fucosyllactose\b", "3-FL"),
        (r"\b6.?SL\b|\b6['-]?sialyllactose\b", "6'-sialyllactose"),
        (r"\bLacto-N-tetraose\b|\bLNT\b", "LNT"),
        (r"\bdifucosyllactose\b|\bDFL\b", "DFL"),
        (r"\bmalic acid\b", "malic acid"),
        (r"\bethanol\b", "ethanol"),
        (r"\berythritol\b", "erythritol"),
        (r"\bxanthohumol\b", "xanthohumol"),
        (r"\bN-acetylneuraminic acid\b|\bNeuAc\b", "N-acetylneuraminic acid"),
        (r"\bhyaluronidase\b", "hyaluronidase"),
        (r"\btrastuzumab Fab\b|\bFab\b", "Fab"),
        (r"\bfull-length antibodies\b", "full-length antibodies"),
        (r"\bD-allulose\b", "D-allulose"),
        (r"\bHMO(?:s)?\b|human milk oligosaccharides", "HMO"),
    ]
    for pattern, replacement in candidates:
        if re.search(pattern, text, flags=re.I):
            return replacement
    return ""


def infer_semantic_type(summary: str, query_type: str) -> str:
    s = summary.lower()
    if not summary:
        return "too_fragmentary"
    if re.search(r"\bcontinued\b", summary, flags=re.I) and len(english_tokens(summary)) < 10:
        return "too_fragmentary"
    if query_type == "normal_control":
        if re.search(r"\bmethod|performed|used to|validate|analysis|ANOVA|SEM|MS/MS|treatment\b", summary, flags=re.I):
            return "normal_method_description"
        return "normal_mechanism_or_result"
    if query_type == "figure_caption":
        if re.search(r"\blocalization|secretion|secretory|tagged protein|exogenous\b", s):
            return "protein_localization_or_activity"
        if re.search(r"\bkinetic|enzyme|activity|selectivity|hydrolysis|variant|wild-type|mutant\b", s):
            return "enzyme_kinetics"
        if re.search(r"\btranscript|expression|upregulated|downregulated|proteomic|genes?|mRNA|efflux pumps?\b", s):
            return "gene_expression_or_transcriptomics"
        if re.search(r"\bgrowth rate|growth curve|growth|assay|challenge|response\b", s):
            return "growth_or_assay_results"
        if re.search(r"\btiters?|yields?|production|productivity|conversion rate|concentrations?|fed-batch|fermentation\b", s):
            return "production_or_titer_results"
        if re.search(r"\bhmo|oligosaccharide|fucosyllactose|sialyllactose|milk group|infant|glycan|gos\b", s):
            return "glycan_or_hmo_composition"
        return "figure_experiment_result"
    if re.search(r"\bprimers?|oligonucleotides?|qRT-PCR|PCR\b", summary, flags=re.I):
        return "primers_oligonucleotides"
    if re.search(r"\bstrains?|plasmids?|vectors?|constructs?|genotype|mutant strains?\b", s):
        return "strains_plasmids"
    if re.search(r"\btranscript|expression|upregulated|downregulated|proteomic|proteins detected|genes?|proteins?|mRNA|ERAD|efflux pumps?\b", s):
        return "gene_expression_or_transcriptomics"
    if re.search(r"\bfermentation|cultivation|culture conditions?|medium|methanol-minimal|shake-flask|bioreactor|process parameters?\b", s):
        return "fermentation_or_cultivation_conditions"
    if re.search(r"\btiters?|yields?|production|productivity|conversion rate|concentrations?|consumption rates?|biomass-specific\b", s):
        return "production_or_titer_results"
    if re.search(r"\bkinetic|kcat|km|enzyme|activity|selectivity|hydrolysis|variant|wild-type|mutant\b", s):
        return "enzyme_kinetics"
    if re.search(r"\bgrowth rate|growth curve|assay|colonised|colonized|challenge|response\b", s):
        return "growth_or_assay_results"
    if re.search(r"\bhmo|oligosaccharide|fucosyllactose|sialyllactose|milk group|infant|glycan|n-glycans?|gos\b", s):
        return "glycan_or_hmo_composition"
    if re.search(r"\blocalization|secretion|secretory|tagged protein|protein activity|extracellular\b", s):
        return "protein_localization_or_activity"
    if query_type == "figure_caption":
        return "figure_experiment_result"
    if parser_artifact_present(summary):
        return "parser_artifact"
    return "other"


def semantic_summary_for_type(summary: str, semantic_type: str) -> str:
    org = organism_phrase(summary)
    product = product_phrase(summary)
    if semantic_type == "strains_plasmids":
        if org and product:
            return f"{org} strains, plasmids, or constructs used for {product} work"
        if org:
            return f"{org} strains, plasmids, vectors, or engineered constructs used in the study"
        if product:
            return f"strains, plasmids, or engineered constructs used for {product} work"
        return "strains, plasmids, vectors, or engineered constructs used in the study"
    if semantic_type == "primers_oligonucleotides":
        if "qrt" in summary.lower():
            return "primer sequences used for qRT-PCR assays"
        if "construction" in summary.lower() or "plasmid" in summary.lower():
            return "primers or oligonucleotides used for plasmid construction"
        return "primers or oligonucleotides used in the study"
    if semantic_type == "fermentation_or_cultivation_conditions":
        if product:
            return f"fermentation or cultivation conditions for {product} production"
        if org:
            return f"cultivation or assay conditions for {org}"
        return "fermentation, cultivation, or assay conditions used in the experiment"
    if semantic_type == "production_or_titer_results":
        if product:
            return f"{product} production titers, yields, or rates compared across conditions"
        return "product titers, yields, or production rates compared across strains or conditions"
    if semantic_type == "enzyme_kinetics":
        return "enzyme activity, selectivity, or kinetic parameters compared across variants"
    if semantic_type == "gene_expression_or_transcriptomics":
        if "bile" in summary.lower() and org:
            return f"genes or proteins upregulated by bile exposure in {org}"
        return "genes or proteins whose expression changed under the tested condition"
    if semantic_type == "growth_or_assay_results":
        return "growth, colonization, assay, or challenge-response results compared across groups"
    if semantic_type == "glycan_or_hmo_composition":
        return "HMO, glycan, milk-group, or oligosaccharide measurements compared across groups"
    if semantic_type == "protein_localization_or_activity":
        return "protein localization, secretion, or activity measured under the study conditions"
    if semantic_type == "figure_experiment_result":
        return figure_topic_summary(summary)
    if semantic_type == "normal_method_description":
        return normal_topic_summary(summary, method=True)
    if semantic_type == "normal_mechanism_or_result":
        return normal_topic_summary(summary, method=False)
    if semantic_type in {"parser_artifact", "too_fragmentary"}:
        return "fragmentary or parser-artifact target text"
    short = remove_table_figure_prefix(summary)
    if len(short) > 160:
        short = short[:160].rsplit(" ", 1)[0]
    return normalize_ws(short)


def figure_topic_summary(summary: str) -> str:
    s = summary.lower()
    if "metal ions" in s or "chemicals" in s:
        return "effect of metal ions or chemicals on enzyme activity"
    if "maldi" in s or "lc-ms" in s or "mass spectrometry" in s or "rp-hplc" in s:
        return "analytical characterization results for the product or protein"
    if "sequence alignment" in s or "docking" in s:
        return "sequence alignment or docking comparison for enzyme variants"
    if "subcellular localization" in s or "localization" in s:
        return "subcellular localization of the tagged or exogenous protein"
    if "growth curve" in s or "growth" in s:
        return "growth comparison for engineered strains"
    if "fermentation" in s or "fed-batch" in s or "production" in s:
        return "fermentation or production results for the engineered strain"
    if "cytokine" in s or "treatment" in s or "challenge" in s:
        return "effect of treatment or challenge conditions on cellular responses"
    if "gene expression" in s or "genes" in s:
        return "gene expression changes under the tested condition"
    if "pathway" in s or "biosynthesis" in s:
        return "biosynthetic pathway or engineering workflow"
    return "main experimental result shown in the figure caption"


def normal_topic_summary(summary: str, method: bool) -> str:
    text = normalize_ws(re.sub(r"^#+\s*\w*:?\s*", "", summary))
    if len(text) > 180:
        text = text[:180].rsplit(" ", 1)[0]
    if method:
        return f"method or validation detail: {text}"
    return f"mechanism or result detail: {text}"


def user_intent_for(row: dict[str, Any], semantic_type: str, semantic_summary: str) -> tuple[str, str, str]:
    query_type = row.get("query_type", "")
    if query_type in {"table_content", "caption_level_table"}:
        return (
            f"find the table that contains or introduces {semantic_summary}",
            "table caption or nearby table-related text",
            "table_locator",
        )
    if query_type == "figure_caption":
        return (
            f"find the figure caption for {semantic_summary}",
            "figure caption text",
            "figure_locator",
        )
    if query_type == "normal_control":
        return (
            f"retrieve the paragraph explaining {semantic_summary}",
            "normal paragraph text",
            "paragraph_fact",
        )
    return (
        f"retrieve evidence for {semantic_summary}",
        "target text",
        "general_retrieval",
    )


def target_flags(row: dict[str, Any], summary: str) -> dict[str, bool]:
    preview = normalize_ws(row.get("target_text_preview", ""))
    query_type = row.get("query_type", "")
    table_caption = bool(re.search(r"\[TABLE\s+CAPTION\]", preview, flags=re.I))
    figure_caption = bool(re.search(r"\[FIGURE\s+CAPTION\]", preview, flags=re.I))
    field_list = target_contains_field_list(summary)
    dense = dense_ids_or_numbers(summary)
    fragment = target_preview_looks_fragment(preview)
    parser_artifact = parser_artifact_present(preview) or parser_artifact_present(summary)
    requires_image = bool(re.search(r"\b(OCR|image-only|read the image|micrograph|gel image)\b", summary, flags=re.I))
    requires_structured_table = bool(
        query_type in {"table_content", "caption_level_table"}
        and field_list
        and dense
        and not re.search(
            r"\b(strains?|plasmids?|primers?|oligonucleotides?|kinetic|activity|genes?|proteins?|growth|yield|production|HMO|milk group)\b",
            summary,
            flags=re.I,
        )
    )
    return {
        "target_is_table_caption_only": table_caption and query_type == "caption_level_table",
        "target_is_table_related_text": query_type in {"table_content", "caption_level_table"} or table_caption,
        "target_is_figure_caption": query_type == "figure_caption" or figure_caption,
        "target_is_normal_paragraph": query_type == "normal_control" and not table_caption and not figure_caption,
        "target_is_fragment_or_parser_artifact": fragment or parser_artifact,
        "target_contains_dense_ids_or_numbers": dense,
        "target_contains_field_list": field_list,
        "target_requires_structured_table": requires_structured_table,
        "target_requires_ocr_or_image": requires_image,
    }


def table_query(summary: str, semantic_type: str) -> tuple[str, str, str]:
    org = organism_phrase(summary)
    product = product_phrase(summary)
    s = summary.lower()
    if semantic_type == "strains_plasmids":
        if org and product:
            return (
                f"Which table describes the {org} strains and plasmids used for {product} work?",
                "high",
                "rewritten from resource-table semantics, not field names",
            )
        if org:
            return (
                f"Which table describes the {org} strains and plasmids used in the study?",
                "high",
                "rewritten from resource-table semantics, not field names",
            )
        if product:
            return (
                f"Where can I find the strains and plasmids used for {product} work?",
                "high",
                "rewritten from resource-table semantics, not field names",
            )
        return (
            "Which table describes the strains and plasmids used in the study?",
            "medium",
            "rewritten from resource-table semantics, not field names",
        )
    if semantic_type == "primers_oligonucleotides":
        if "qrt" in s:
            return (
                "Which table gives the primer sequences used for qRT-PCR?",
                "high",
                "rewritten to a natural primer-table lookup",
            )
        if "construction" in s or "plasmid" in s:
            return (
                "Which table lists primers used for plasmid construction?",
                "high",
                "rewritten to a natural primer-table lookup",
            )
        return (
            "Which table lists primers or oligonucleotides used in the study?",
            "medium",
            "rewritten to a natural primer-table lookup",
        )
    if semantic_type == "fermentation_or_cultivation_conditions":
        if product:
            return (
                f"Which table reports cultivation conditions for {product} production?",
                "high",
                "rewritten to a natural conditions-table lookup",
            )
        return (
            "Which table reports culture or assay conditions used in the experiment?",
            "medium",
            "rewritten to a natural conditions-table lookup",
        )
    if semantic_type == "production_or_titer_results":
        if "ethanol" in s and ("xylose" in s or "glucose" in s or "arabinose" in s):
            return (
                "Which table compares ethanol yields and sugar consumption rates?",
                "high",
                "rewritten to a natural production-results lookup",
            )
        if product and product != "HMO":
            return (
                f"Where are {product} titers or yields compared across engineered strains?",
                "high",
                "rewritten to a natural production-results lookup",
            )
        return (
            "Where are product titers or yields compared across engineered strains?",
            "medium",
            "rewritten to a natural production-results lookup",
        )
    if semantic_type == "enzyme_kinetics":
        return (
            "Which table compares enzyme activity or kinetic parameters across variants?",
            "high",
            "rewritten to a natural enzyme-parameter lookup",
        )
    if semantic_type == "gene_expression_or_transcriptomics":
        if "proteins detected" in s and "culture media" in s:
            org = organism_phrase(summary) or "P. pastoris"
            return (
                f"Which table lists proteins detected in {org} culture media?",
                "high",
                "rewritten to a natural proteomics-table lookup",
            )
        if "bile" in s and org:
            return (
                f"Where can I find genes and proteins upregulated by bile exposure in {org}?",
                "high",
                "rewritten to a natural gene-expression lookup",
            )
        if "erad" in s:
            return (
                "Which table summarizes gene disruption targets related to ERAD?",
                "high",
                "rewritten to a natural gene-target lookup",
            )
        return (
            "Which table lists genes or proteins whose expression changed under the tested condition?",
            "medium",
            "rewritten to a natural gene-expression lookup",
        )
    if semantic_type == "growth_or_assay_results":
        return (
            "Which table compares growth or assay results across the tested groups?",
            "medium",
            "rewritten to a natural growth/assay lookup",
        )
    if semantic_type == "glycan_or_hmo_composition":
        return (
            "Which table compares HMO measurements across milk or infant groups?",
            "medium",
            "rewritten to a natural HMO-table lookup",
        )
    if semantic_type == "protein_localization_or_activity":
        return (
            "Which table reports protein localization, secretion, or activity measurements?",
            "medium",
            "rewritten to a natural protein-measurement lookup",
        )
    return ("", "low", "no reliable natural table question from target semantics")


def figure_query(summary: str) -> tuple[str, str, str]:
    s = summary.lower()
    if "metal ions" in s or "chemicals" in s:
        return (
            "Which figure shows how metal ions affect EntFuc hydrolysis activity?",
            "high",
            "rewritten to a natural figure-caption lookup",
        )
    if "maldi" in s or "lc-ms" in s or "mass spectrometry" in s:
        return (
            "Which figure presents analytical characterization of the protein products?",
            "high",
            "rewritten to a natural figure-caption lookup",
        )
    if "sequence alignment" in s or "docking" in s:
        return (
            "Which figure compares sequence alignment or docking results for enzyme variants?",
            "high",
            "rewritten to a natural figure-caption lookup",
        )
    if "subcellular localization" in s or "localization" in s:
        return (
            "Which figure illustrates subcellular localization of the tagged protein?",
            "high",
            "rewritten to a natural figure-caption lookup",
        )
    if "growth curve" in s or "growth" in s:
        return (
            "Which figure compares growth across the engineered strains?",
            "high",
            "rewritten to a natural figure-caption lookup",
        )
    if "fed-batch" in s or "fermentation" in s or "production" in s:
        product = product_phrase(summary)
        if product:
            return (
                f"Which figure shows fermentation results for {product} production?",
                "high",
                "rewritten to a natural figure-caption lookup",
            )
        return (
            "Which figure compares growth or production across engineered strains?",
            "medium",
            "rewritten to a natural figure-caption lookup",
        )
    if "cytokine" in s or "treatment" in s or "challenge" in s:
        return (
            "Which figure shows the effect of treatment on cellular responses?",
            "medium",
            "rewritten to a natural figure-caption lookup",
        )
    if "gene expression" in s or "genes" in s:
        return (
            "Which figure compares gene expression changes under the tested condition?",
            "medium",
            "rewritten to a natural figure-caption lookup",
        )
    if "pathway" in s or "biosynthesis" in s:
        return (
            "Which figure illustrates the biosynthetic pathway used in the study?",
            "medium",
            "rewritten to a natural figure-caption lookup",
        )
    return (
        "Which figure shows the main experimental result for the tested condition?",
        "low",
        "generic figure-caption rewrite because target semantics were broad",
    )


def normalize_normal_query(row: dict[str, Any], used_queries: set[str]) -> tuple[str, str, str]:
    sample_id = row.get("sample_id", "")
    current = normalize_ws(row.get("query", ""))
    if sample_id in NORMAL_QUERY_OVERRIDES:
        return (
            NORMAL_QUERY_OVERRIDES[sample_id],
            "high",
            "rewritten to remove duplicate, vague, or short normal-control wording",
        )
    issues = hard_rule_lint(current, row.get("target_text_preview", ""), "normal_control")
    if current.lower() in used_queries or issues:
        preview = normalize_ws(row.get("target_text_preview", ""))
        if "PNGase F" in preview:
            return (
                "Why was PNGase F treatment performed before MS/MS analysis?",
                "high",
                "rewritten to satisfy strict length and specificity rules",
            )
        if "ANOVA" in preview:
            return (
                "What group differences were compared using one-way ANOVA?",
                "high",
                "rewritten to satisfy strict length and specificity rules",
            )
        if "SEM" in preview:
            return (
                "What did SEM analysis investigate in the treated material?",
                "medium",
                "rewritten to satisfy strict length and specificity rules",
            )
    return current, "high", "original normal-control query is already natural"


def propose_query(row: dict[str, Any], summary: str, semantic_type: str, used_normal_queries: set[str]) -> tuple[str, str, str]:
    query_type = row.get("query_type", "")
    if query_type == "normal_control":
        return normalize_normal_query(row, used_normal_queries)
    if query_type in {"table_content", "caption_level_table"}:
        return table_query(summary, semantic_type)
    if query_type == "figure_caption":
        return figure_query(summary)
    return ("", "low", "unsupported query type for strict main rewrite")


def natural_feasibility(
    row: dict[str, Any],
    semantic_type: str,
    flags: dict[str, bool],
    rewritten_query: str,
    final_issues: list[str],
    confidence: str,
) -> tuple[bool, str, str, bool, bool, bool, bool]:
    query_type = row.get("query_type", "")
    if not stable_ids(row):
        return False, "none", "stable_target_block_ids missing", False, False, True, False
    if flags["target_requires_ocr_or_image"]:
        return False, "none", "target requires OCR or image understanding", False, False, True, False
    if flags["target_requires_structured_table"]:
        return False, "none", "target requires row/cell structured table access", False, True, True, False
    if flags["target_is_fragment_or_parser_artifact"]:
        if (
            semantic_type in {"parser_artifact", "too_fragmentary", "other"}
            and row.get("sample_id") not in METADATA_SCOPE_REVIEW_OVERRIDES
        ):
            return False, "none", "target is fragmentary or parser-artifact text", False, False, True, True
    if not rewritten_query:
        return False, "none", "no reliable natural rewrite", False, True, False, True
    if final_issues:
        if query_type in {"table_content", "caption_level_table", "figure_caption"}:
            return False, "lexical_or_caption_anchor", "rewrite fails hard-rule lint", False, True, False, False
        return False, "paragraph_fact", "normal-control query fails hard-rule lint", False, False, True, True
    if confidence == "low" and query_type != "normal_control":
        return False, "lexical_or_caption_anchor", "rewrite confidence is too low for strict main", False, True, False, False
    if (
        semantic_type == "other"
        and query_type in {"table_content", "caption_level_table"}
        and row.get("sample_id") not in METADATA_SCOPE_REVIEW_OVERRIDES
    ):
        return False, "lexical_or_caption_anchor", "target semantics are too broad for strict main", False, True, False, True
    if query_type == "normal_control":
        return True, "paragraph_fact", "", True, False, False, False
    if query_type == "figure_caption":
        return True, "figure_locator", "", True, False, False, False
    return True, "table_locator", "", True, False, False, False


def evaluate_main_row(
    row: dict[str, Any],
    chunks: dict[str, dict[str, str]],
    used_normal_queries: set[str],
) -> dict[str, Any]:
    current_query = normalize_ws(row.get("query", ""))
    original_query = normalize_ws(row.get("original_query") or current_query)
    summary_raw = target_summary_text(row, chunks)
    semantic_type = infer_semantic_type(summary_raw, row.get("query_type", ""))
    flags = target_flags(row, summary_raw)
    if flags["target_is_fragment_or_parser_artifact"] and semantic_type == "other":
        semantic_type = "parser_artifact" if parser_artifact_present(summary_raw) else "too_fragmentary"
    semantic_summary = semantic_summary_for_type(summary_raw, semantic_type)
    user_intent, answer_anchor, natural_type_default = user_intent_for(row, semantic_type, semantic_summary)
    metadata_override = METADATA_SCOPE_REVIEW_OVERRIDES.get(str(row.get("sample_id", "")))
    if metadata_override:
        semantic_type = metadata_override["target_semantic_type"]
        semantic_summary = metadata_override["target_semantic_summary"]
        user_intent = metadata_override["user_intent"]
        answer_anchor = metadata_override["answer_anchor"]
    rewritten_query, confidence, rewrite_reason = propose_query(
        row, summary_raw, semantic_type, used_normal_queries
    )
    if row.get("sample_id") in TABLE_CONTENT_SCOPE_REVIEW_QUERY_OVERRIDES:
        rewritten_query = TABLE_CONTENT_SCOPE_REVIEW_QUERY_OVERRIDES[str(row.get("sample_id"))]
        confidence = "high"
        rewrite_reason = "manual table_content scope review matched the query to answerable table semantics"
    if row.get("sample_id") in CAPTION_SCOPE_REVIEW_QUERY_OVERRIDES:
        rewritten_query = CAPTION_SCOPE_REVIEW_QUERY_OVERRIDES[str(row.get("sample_id"))]
        confidence = "high"
        rewrite_reason = "manual caption scope review matched the query to answerable caption semantics"
    if row.get("sample_id") in CAPTION_SCOPE_REVIEW_DEMOTE_TO_LEXICAL:
        rewritten_query = CAPTION_SCOPE_REVIEW_DEMOTE_TO_LEXICAL[str(row.get("sample_id"))]
        confidence = "high"
        rewrite_reason = "manual caption scope review demoted weak supplementary pointer from strict main"
    if row.get("sample_id") in TABLE_CONTENT_SCOPE_REVIEW_DEMOTE_TO_LEXICAL:
        rewritten_query = TABLE_CONTENT_SCOPE_REVIEW_DEMOTE_TO_LEXICAL[str(row.get("sample_id"))]
        confidence = "high"
        rewrite_reason = "manual table_content scope review demoted duplicate or weak pointer from strict main"
    current_issues = hard_rule_lint(current_query, row.get("target_text_preview", ""), row.get("query_type", ""))
    final_issues = hard_rule_lint(rewritten_query, row.get("target_text_preview", ""), row.get("query_type", "")) if rewritten_query else ["no_rewrite_candidate"]
    (
        can_form,
        natural_type,
        reason_if_not,
        should_main,
        should_lexical,
        should_diag,
        should_manual,
    ) = natural_feasibility(row, semantic_type, flags, rewritten_query, final_issues, confidence)

    if row.get("sample_id") in CAPTION_SCOPE_REVIEW_DEMOTE_TO_LEXICAL or row.get("sample_id") in TABLE_CONTENT_SCOPE_REVIEW_DEMOTE_TO_LEXICAL:
        decision = "move_to_lexical_stress"
        final_label = "lexical_stress"
        final_query = rewritten_query or current_query
    elif can_form and should_main:
        if normalize_ws(rewritten_query) == current_query and not current_issues:
            decision = "keep_original"
            final_label = "main_strict"
            final_query = current_query
        else:
            decision = "rewrite_and_keep_main"
            final_label = "main_rewritten"
            final_query = rewritten_query
    elif should_lexical:
        decision = "move_to_lexical_stress"
        final_label = "lexical_stress"
        final_query = rewritten_query or current_query
    elif should_manual:
        decision = "manual_review"
        final_label = "needs_manual_review"
        final_query = rewritten_query or current_query
    elif should_diag:
        decision = "move_to_diagnostic"
        final_label = "diagnostic_active"
        final_query = rewritten_query or current_query
    else:
        decision = "exclude_from_current_eval"
        final_label = "exclude_from_current_eval"
        final_query = rewritten_query or current_query

    rationale_bits = []
    if reason_if_not:
        rationale_bits.append(reason_if_not)
    if flags["target_contains_dense_ids_or_numbers"]:
        rationale_bits.append("target has dense IDs or numbers")
    if flags["target_contains_field_list"]:
        rationale_bits.append("target has field-list/table-header content")
    if current_issues:
        rationale_bits.append(f"current query lint: {issue_join(current_issues)}")
    if final_issues and final_label not in {"main_strict", "main_rewritten"}:
        rationale_bits.append(f"rewrite lint: {issue_join(final_issues)}")
    if not rationale_bits:
        rationale_bits.append(rewrite_reason)

    audit = {
        "sample_id": row.get("sample_id", ""),
        "query_type": row.get("query_type", ""),
        "original_query": original_query,
        "current_query": current_query,
        "target_doc_id": row.get("target_doc_id", ""),
        "stable_target_block_ids": stable_ids_s(row),
        "target_text_preview": row.get("target_text_preview", ""),
        "target_semantic_type": semantic_type,
        "target_semantic_summary": semantic_summary,
        "user_intent": user_intent,
        "answer_anchor": answer_anchor,
        "can_form_natural_question": str(can_form).lower(),
        "natural_question_type": natural_type or natural_type_default,
        "reason_if_not_natural": reason_if_not,
        "should_be_main_strict": str(should_main).lower(),
        "should_be_lexical_stress": str(should_lexical).lower(),
        "should_be_diagnostic": str(should_diag).lower(),
        "should_need_manual_review": str(should_manual).lower(),
        "hard_rule_violations": issue_join(final_issues if final_issues else current_issues),
        "current_hard_rule_violations": issue_join(current_issues),
        "final_hard_rule_violations": issue_join(final_issues),
        "semantic_decision": decision,
        "rewritten_query": final_query,
        "rewrite_confidence": confidence,
        "final_label": final_label,
        "rationale": issue_join(rationale_bits),
        "rewrite_reason": rewrite_reason,
        "target_semantic_summary_raw": summary_raw,
        **{key: str(value).lower() for key, value in flags.items()},
    }
    return audit


def strict_row_from_audit(row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    final_query = normalize_ws(audit["rewritten_query"])
    previous_query = normalize_ws(row.get("query", ""))
    return {
        "sample_id": row.get("sample_id", ""),
        "original_sample_id": row.get("original_sample_id") or row.get("sample_id", ""),
        "query_type": row.get("query_type", ""),
        "query": final_query,
        "original_query": normalize_ws(row.get("original_query") or previous_query),
        "previous_query": previous_query,
        "query_was_rewritten": final_query != previous_query or bool(row.get("query_was_rewritten")),
        "target_doc_id": row.get("target_doc_id", ""),
        "stable_target_block_ids": stable_ids(row),
        "target_caption_block_id": row.get("target_caption_block_id", ""),
        "target_associated_block_id": row.get("target_associated_block_id", ""),
        "target_chunk_id_candidate": row.get("target_chunk_id_candidate", ""),
        "source_phase": row.get("source_phase", ""),
        "source_file": row.get("source_file", ""),
        "ability_scope": row.get("ability_scope") or ability_scope_for(row.get("query_type", "")),
        "expected_capability": row.get("expected_capability") or expected_capability_for(row.get("query_type", "")),
        "target_semantic_type": audit["target_semantic_type"],
        "target_semantic_summary": audit["target_semantic_summary"],
        "user_intent": audit["user_intent"],
        "semantic_quality_label": audit["final_label"],
        "hard_rule_passed": True,
        "include_in_main_denominator": True,
        "target_text_preview": clean_preview_for_output(row.get("target_text_preview", "")),
        "rationale": audit["rationale"],
        "notes": "phase5f_3c_strict_semantic_enhancement",
    }


def ability_scope_for(query_type: str) -> str:
    if query_type in {"table_content", "caption_level_table"}:
        return "table_related_text_retrieval"
    if query_type == "figure_caption":
        return "figure_caption_retrieval"
    if query_type == "normal_control":
        return "normal_paragraph_retrieval"
    return "retrieval"


def expected_capability_for(query_type: str) -> str:
    if query_type in {"table_content", "caption_level_table"}:
        return "retrieve table-related text associated with stable caption or nearby table blocks"
    if query_type == "figure_caption":
        return "retrieve figure caption text without visual interpretation"
    if query_type == "normal_control":
        return "retrieve normal paragraph evidence"
    return "retrieve target evidence"


def process_normal_recovery(
    normal_rows: list[dict[str, Any]],
    chunks: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    used_queries: set[str] = set()
    forbidden_flags = {
        "title_derived_or_mechanical",
        "table_like_not_normal",
        "query_target_mismatch",
        "retrieval_issue_candidate",
        "needs_manual_review",
    }
    for row in normal_rows:
        risk_flags_raw = row.get("quality_risk_flags") or []
        if isinstance(risk_flags_raw, str):
            risk_flags = {risk_flags_raw} if risk_flags_raw else set()
        else:
            risk_flags = {str(flag) for flag in risk_flags_raw}
        rewritten, confidence, reason = normalize_normal_query(row, used_queries)
        issues = hard_rule_lint(rewritten, row.get("target_text_preview", ""), "normal_control")
        selected_for_main = (
            len(selected) < 30
            and row.get("quality_label") == "good_normal_control"
            and bool(stable_ids(row))
            and not risk_flags.intersection(forbidden_flags)
            and not issues
        )
        decision = "selected" if selected_for_main else "rejected"
        rationale = reason
        if risk_flags.intersection(forbidden_flags):
            rationale = f"forbidden risk flags: {issue_join(sorted(risk_flags.intersection(forbidden_flags)))}"
        if issues:
            rationale = f"hard-rule violations: {issue_join(issues)}"
        if not stable_ids(row):
            rationale = "missing stable_target_block_ids"
        audit = {
            "sample_id": row.get("sample_id", ""),
            "query": row.get("query", ""),
            "rewritten_query": rewritten,
            "query_was_rewritten": str(rewritten != normalize_ws(row.get("query", ""))).lower(),
            "target_doc_id": row.get("target_doc_id", ""),
            "stable_target_block_ids": stable_ids_s(row),
            "quality_label": row.get("quality_label", ""),
            "quality_risk_flags": "|".join(sorted(risk_flags)),
            "hard_rule_violations": issue_join(issues),
            "selected_for_main": str(selected_for_main).lower(),
            "recovery_decision": decision,
            "rationale": rationale,
        }
        audit_rows.append(audit)
        if selected_for_main:
            used_queries.add(rewritten.lower())
            selected.append(row)
            synthetic_audit = evaluate_main_row(
                {**row, "query": rewritten, "original_query": row.get("query", "")},
                chunks,
                used_queries,
            )
            synthetic_audit["rewritten_query"] = rewritten
            synthetic_audit["final_hard_rule_violations"] = ""
            synthetic_audit["hard_rule_violations"] = ""
            synthetic_audit["semantic_decision"] = (
                "rewrite_and_keep_main" if rewritten != normalize_ws(row.get("query", "")) else "keep_original"
            )
            synthetic_audit["final_label"] = (
                "main_rewritten" if rewritten != normalize_ws(row.get("query", "")) else "main_strict"
            )
            strict_rows.append(strict_row_from_audit({**row, "query": rewritten}, synthetic_audit))
    return selected, audit_rows, strict_rows


def dedupe_and_cap_strict(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kept: list[dict[str, Any]] = []
    demotions: list[dict[str, str]] = []
    seen_sample: set[str] = set()
    seen_full: set[str] = set()
    exact_query_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    resource_semantic_count = 0
    doc_counts: Counter[str] = Counter()
    for row in rows:
        sample_id = row.get("sample_id", "")
        full_key = f"{row.get('query','').lower()}|{row.get('target_doc_id','')}|{'|'.join(row.get('stable_target_block_ids') or [])}"
        query_key = row.get("query", "").lower()
        semantic_type = row.get("target_semantic_type", "")
        query_type = row.get("query_type", "")
        reason = ""
        if sample_id in seen_sample:
            reason = "duplicate_sample_id"
        elif full_key in seen_full:
            reason = "duplicate_query_doc_stable_target"
        elif query_type != "normal_control" and exact_query_counts[query_key] >= 3:
            reason = "exact_query_template_cap"
        elif query_type != "normal_control" and doc_counts[row.get("target_doc_id", "")] >= 6:
            reason = "doc_contribution_cap"
        elif (
            query_type != "normal_control"
            and semantic_type in {"strains_plasmids", "primers_oligonucleotides"}
            and resource_semantic_count >= 30
        ):
            reason = "resource_question_distribution_cap"
        elif query_type != "normal_control" and semantic_counts[semantic_type] >= 36:
            reason = "semantic_type_distribution_cap"
        if reason:
            demotions.append(
                {
                    "sample_id": sample_id,
                    "query": row.get("query", ""),
                    "query_type": query_type,
                    "target_doc_id": row.get("target_doc_id", ""),
                    "stable_target_block_ids": "|".join(row.get("stable_target_block_ids") or []),
                    "target_text_preview": row.get("target_text_preview", ""),
                    "demotion_reason": reason,
                }
            )
            continue
        kept.append(row)
        seen_sample.add(sample_id)
        seen_full.add(full_key)
        exact_query_counts[query_key] += 1
        semantic_counts[semantic_type] += 1
        if query_type != "normal_control" and semantic_type in {"strains_plasmids", "primers_oligonucleotides"}:
            resource_semantic_count += 1
        doc_counts[row.get("target_doc_id", "")] += 1
    return kept, demotions


def lexical_row_from_audit(row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id", ""),
        "original_sample_id": row.get("original_sample_id") or row.get("sample_id", ""),
        "query_type": row.get("query_type", ""),
        "query": audit.get("rewritten_query") or row.get("query", ""),
        "original_query": row.get("original_query") or row.get("query", ""),
        "previous_query": row.get("query", ""),
        "include_in_main_denominator": False,
        "diagnostic_label": "lexical_stress",
        "target_doc_id": row.get("target_doc_id", ""),
        "stable_target_block_ids": stable_ids(row),
        "target_caption_block_id": row.get("target_caption_block_id", ""),
        "target_associated_block_id": row.get("target_associated_block_id", ""),
        "target_chunk_id_candidate": row.get("target_chunk_id_candidate", ""),
        "target_semantic_type": audit.get("target_semantic_type", ""),
        "target_semantic_summary": audit.get("target_semantic_summary", ""),
        "target_text_preview": row.get("target_text_preview", ""),
        "source_phase": row.get("source_phase", ""),
        "source_file": row.get("source_file", ""),
        "demotion_reason": audit.get("rationale", ""),
        "hard_rule_violations": audit.get("hard_rule_violations", ""),
    }


def diagnostic_row_from_audit(row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    final_label = audit.get("final_label", "")
    return {
        "sample_id": row.get("sample_id", ""),
        "original_sample_id": row.get("original_sample_id") or row.get("sample_id", ""),
        "query_type": row.get("query_type", ""),
        "query": audit.get("rewritten_query") or row.get("query", ""),
        "original_query": row.get("original_query") or row.get("query", ""),
        "previous_query": row.get("query", ""),
        "include_in_main_denominator": False,
        "diagnostic_label": final_label,
        "target_doc_id": row.get("target_doc_id", ""),
        "stable_target_block_ids": stable_ids(row),
        "target_caption_block_id": row.get("target_caption_block_id", ""),
        "target_associated_block_id": row.get("target_associated_block_id", ""),
        "target_chunk_id_candidate": row.get("target_chunk_id_candidate", ""),
        "target_semantic_type": audit.get("target_semantic_type", ""),
        "target_semantic_summary": audit.get("target_semantic_summary", ""),
        "target_text_preview": row.get("target_text_preview", ""),
        "source_phase": row.get("source_phase", ""),
        "source_file": row.get("source_file", ""),
        "demotion_reason": audit.get("rationale", ""),
        "hard_rule_violations": audit.get("hard_rule_violations", ""),
    }


def previous_lexical_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["include_in_main_denominator"] = False
    out["diagnostic_label"] = "lexical_stress"
    out["previous_query"] = row.get("query", "")
    out["demotion_reason"] = row.get("rationale") or row.get("detected_issues") or "previous lexical stress row"
    return out


def template_key(query: str) -> str:
    q = normalize_ws(query).lower().rstrip("?")
    q = re.sub(r"\b(?:e\. coli|s\. cerevisiae|p\. pastoris|k\. phaffii|b\. subtilis|l\. casei|b\. longum)\b", "<organism>", q)
    q = re.sub(r"\b(?:3'-sialyllactose|2'-fucosyllactose|3-fl|6'-sialyllactose|lnt|dfl|hmo|ethanol|malic acid|erythritol)\b", "<product>", q)
    q = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", q)
    q = re.sub(r"\b[A-Z][A-Za-z0-9'/-]*\b", "<id>", q)
    return q


def distribution_bins(lengths: list[int]) -> dict[str, int]:
    bins = {"lt8": 0, "8_16": 0, "17_24": 0, "25_28": 0, "gt28": 0}
    for length in lengths:
        if length < 8:
            bins["lt8"] += 1
        elif length <= 16:
            bins["8_16"] += 1
        elif length <= 24:
            bins["17_24"] += 1
        elif length <= 28:
            bins["25_28"] += 1
        else:
            bins["gt28"] += 1
    return bins


def make_duplicate_audit(strict_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    dup_rows: list[dict[str, str]] = []

    def add_group(check_type: str, key: str, rows: list[dict[str, Any]], notes: str) -> None:
        dup_rows.append(
            {
                "check_type": check_type,
                "key": key,
                "count": str(len(rows)),
                "sample_ids": "|".join(str(r.get("sample_id", "")) for r in rows[:30]),
                "query_types": "|".join(sorted({str(r.get("query_type", "")) for r in rows})),
                "target_doc_ids": "|".join(sorted({str(r.get("target_doc_id", "")) for r in rows})),
                "notes": notes,
            }
        )

    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        groups[normalize_ws(row.get("query", "")).lower()].append(row)
    for key, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(rows) > 1:
            add_group("same_query_multiple_targets", key, rows, "same query text appears with multiple targets")

    groups.clear()
    for row in strict_rows:
        groups[row_target_key(row)].append(row)
    for key, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(rows) > 1:
            add_group("same_target_multiple_queries", key, rows, "same target appears with multiple queries")

    groups.clear()
    for row in strict_rows:
        groups[template_key(row.get("query", ""))].append(row)
    for key, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(rows) >= 4:
            add_group("query_template_repeated", key, rows, "query template repeated four or more times")

    groups.clear()
    for row in strict_rows:
        groups[str(row.get("target_doc_id", ""))].append(row)
    for key, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(rows) >= 5:
            add_group("doc_overrepresented", key, rows, "document contributes five or more strict samples")

    violation_counts = Counter()
    for row in audit_rows:
        for issue in str(row.get("current_hard_rule_violations", "")).split("; "):
            if issue:
                violation_counts[issue] += 1
    for issue, count in violation_counts.most_common():
        dup_rows.append(
            {
                "check_type": "input_hard_rule_violation",
                "key": issue,
                "count": str(count),
                "sample_ids": "",
                "query_types": "",
                "target_doc_ids": "",
                "notes": "violation counted on current input main query",
            }
        )
    return dup_rows


def write_schema_docs() -> None:
    write_text(
        OUT_DIR / "semantic_decision_schema.md",
        """# Semantic Decision Schema

Each input sample is processed in three ordered steps. A sample can enter
strict main only when Step 2 says it can form a natural question and the final
rewritten query passes the hard-rule lint.

## Step 1: target_semantic_abstraction

The target is abstracted into user intent rather than compressed from a
caption. Fields:

- target_semantic_type
- target_semantic_summary
- user_intent
- answer_anchor
- target_is_table_caption_only
- target_is_table_related_text
- target_is_figure_caption
- target_is_normal_paragraph
- target_is_fragment_or_parser_artifact
- target_contains_dense_ids_or_numbers
- target_contains_field_list
- target_requires_structured_table
- target_requires_ocr_or_image

Allowed target_semantic_type values:

- strains_plasmids
- primers_oligonucleotides
- fermentation_or_cultivation_conditions
- production_or_titer_results
- enzyme_kinetics
- gene_expression_or_transcriptomics
- growth_or_assay_results
- glycan_or_hmo_composition
- protein_localization_or_activity
- figure_experiment_result
- normal_method_description
- normal_mechanism_or_result
- parser_artifact
- too_fragmentary
- other

## Step 2: natural_question_feasibility

Fields:

- can_form_natural_question
- natural_question_type
- reason_if_not_natural
- should_be_main_strict
- should_be_lexical_stress
- should_be_diagnostic
- should_need_manual_review

Only samples with can_form_natural_question=true and
should_be_main_strict=true can enter strict main.

## Step 3: rewrite_or_demote

Allowed decisions:

- keep_original
- rewrite_and_keep_main
- move_to_lexical_stress
- move_to_diagnostic
- manual_review
- exclude_from_current_eval
""",
    )
    write_text(
        OUT_DIR / "hard_rule_lint_spec.md",
        """# Hard-Rule Lint Spec

Strict main queries must pass all hard rules below.

1. Artifact ban: no CAPTION, TABLE CAPTION, FIGURE CAPTION, DOC/DOCX,
   supplementary-material artifact wording, table/figure stubs, primary
   "continued" content, or parser line-break artifacts.
2. Length: English queries should be 8-24 words. Queries above 28 words are
   rejected; 25-28 words require manual justification and are not used here.
   Chinese queries must be natural complete questions, not headers or OCR
   fragments.
3. Copying: no more than seven consecutive words copied from target preview;
   no caption-compression query; no direct table-header field string.
4. ID/number density: more than four ID/gene/strain/plasmid/sequence tokens,
   more than three numeric tokens, long sequence/primer tokens, or very high
   uppercase abbreviation density fails strict main.
5. Template ban: no "Which table summarizes + long caption", table-header
   token soup, "Where are X Y Z summarized in a table", dense figure-caption
   token soup, or "What does Table/Figure N report" forms.
6. Grammar: query must be a complete natural question and not caption syntax
   such as "Which table summarizes ... used in This Study?".
7. Ability boundary: no row/cell-level structured-table question, OCR/image
   question, visual-only figure interpretation, or table_object/figure_object
   dependency.
8. Distribution checks: exact query templates, strains/plasmids and primer
   questions, target documents, and target_semantic_type distributions are
   audited after generation; excessive repeats are demoted before final output.
""",
    )


def write_markdown_reports(
    main_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    lexical_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    normal_audit: list[dict[str, Any]],
    duplicate_rows: list[dict[str, str]],
    missing_inputs: list[str],
    distribution: dict[str, Any],
) -> None:
    qt_counts = Counter(row.get("query_type", "") for row in strict_rows)
    sem_counts = Counter(row.get("target_semantic_type", "") for row in strict_rows)
    final_counts = Counter(row.get("final_label", "") for row in audit_rows)
    current_violation_counts = Counter()
    for row in audit_rows:
        for issue in str(row.get("current_hard_rule_violations", "")).split("; "):
            if issue:
                current_violation_counts[issue] += 1
    strict_violation_counts = Counter()
    for row in strict_rows:
        for issue in hard_rule_lint(row.get("query", ""), row.get("target_text_preview", ""), row.get("query_type", "")):
            strict_violation_counts[issue] += 1

    normal_selected = sum(1 for row in strict_rows if row.get("query_type") == "normal_control")
    rewrite_count = sum(1 for row in strict_rows if row.get("query_was_rewritten"))
    kept_count = len(strict_rows) - rewrite_count
    stable_count = sum(1 for row in strict_rows if row.get("stable_target_block_ids"))
    strict_caption_residue = sum(
        1
        for row in strict_rows
        if re.search(r"\bCAPTION\b", row.get("query", ""), flags=re.I)
        or re.search(r"\bCAPTION\b", row.get("target_text_preview", ""), flags=re.I)
    )
    strict_target_chunk_id_only = sum(1 for row in strict_rows if "target_chunk_id_only" in row)
    strict_needs_manual = sum(1 for row in strict_rows if row.get("semantic_quality_label") == "needs_manual_review")
    strict_ability_violations = sum(
        1
        for row in strict_rows
        if hard_rule_lint(row.get("query", ""), row.get("target_text_preview", ""), row.get("query_type", ""))
    )
    lengths = [len(english_tokens(row.get("query", ""))) for row in strict_rows]
    length_bins = distribution_bins(lengths)
    copy_risk = sum(
        1 for row in strict_rows if max_shared_ngram(row.get("query", ""), row.get("target_text_preview", "")) > 7
    )
    id_risk = sum(
        1
        for row in strict_rows
        if sum(1 for token in english_tokens(row.get("query", "")) if technical_token(token)) > 4
        or numeric_count(english_tokens(row.get("query", ""))) > 3
    )
    recommend_phase5f4 = not strict_violation_counts and stable_count == len(strict_rows) and len(strict_rows) > 0

    audit_md_lines = [
        "# Semantic Enhancement Audit",
        "",
        f"- input_main_count: {len(main_rows)}",
        f"- strict_main_v2_count: {len(strict_rows)}",
        f"- lexical_stress_from_current_main: {final_counts.get('lexical_stress', 0)}",
        f"- diagnostic/manual/exclude_from_current_main: {final_counts.get('diagnostic_active', 0) + final_counts.get('needs_manual_review', 0) + final_counts.get('exclude_from_current_eval', 0)}",
        "",
        "## Final Labels",
    ]
    audit_md_lines.extend(f"- {label}: {count}" for label, count in sorted(final_counts.items()))
    audit_md_lines.extend(["", "## Current Query Hard-Rule Violations"])
    if current_violation_counts:
        audit_md_lines.extend(f"- {issue}: {count}" for issue, count in current_violation_counts.most_common())
    else:
        audit_md_lines.append("- none")
    write_text(OUT_DIR / "semantic_enhancement_audit.md", "\n".join(audit_md_lines))

    rewrite_examples = []
    for row in audit_rows:
        if row.get("semantic_decision") == "rewrite_and_keep_main":
            rewrite_examples.append(row)
        if len(rewrite_examples) >= 25:
            break
    write_text(
        OUT_DIR / "rewrite_examples_v2.md",
        "\n".join(
            ["# Rewrite Examples", ""]
            + [
                "\n".join(
                    [
                        f"## {row['sample_id']}",
                        f"- original: {row['current_query']}",
                        f"- rewritten: {row['rewritten_query']}",
                        f"- intent: {row['user_intent']}",
                        f"- reason: {row['rewrite_reason']}",
                    ]
                )
                for row in rewrite_examples
            ]
        ),
    )

    write_text(
        OUT_DIR / "strict_main_eval_set_v2_summary.md",
        "\n".join(
            [
                "# Strict Main Eval Set V2 Summary",
                "",
                f"- strict_main_v2_count: {len(strict_rows)}",
                f"- rewritten_count: {rewrite_count}",
                f"- kept_original_count: {kept_count}",
                f"- stable_target_coverage: {stable_count}/{len(strict_rows)}",
                f"- hard_rule_violations_in_strict: {sum(strict_violation_counts.values())}",
                f"- caption_residue_in_strict_fields: {strict_caption_residue}",
                f"- normal_control_count: {normal_selected}",
                "",
                "## Query Type Distribution",
                *[f"- {key}: {value}" for key, value in sorted(qt_counts.items())],
                "",
                "## Target Semantic Type Distribution",
                *[f"- {key}: {value}" for key, value in sem_counts.most_common()],
            ]
        ),
    )

    write_text(
        OUT_DIR / "normal_recovery_summary.md",
        "\n".join(
            [
                "# Normal Recovery Summary",
                "",
                f"- good_normal_control_input_count: {len(normal_audit)}",
                f"- selected_for_strict_main: {sum(1 for row in normal_audit if row.get('selected_for_main') == 'true')}",
                f"- final_normal_control_count: {normal_selected}",
                f"- target_normal_control_goal_met: {str(normal_selected == 30).lower()}",
                "",
                "Rows were accepted only from good_normal_control_merged.jsonl with stable targets, no forbidden risk flags, and no hard-rule lint violations.",
            ]
        ),
    )

    write_text(
        OUT_DIR / "lexical_stress_summary_v2.md",
        "\n".join(
            [
                "# Lexical Stress Eval Set V2 Summary",
                "",
                f"- lexical_stress_v2_count: {len(lexical_rows)}",
                f"- demoted_from_current_main: {final_counts.get('lexical_stress', 0)}",
                f"- previous_lexical_rows_included: {max(0, len(lexical_rows) - final_counts.get('lexical_stress', 0))}",
                "",
                "Lexical stress rows are excluded from the main denominator and should be reported separately.",
            ]
        ),
    )

    manual_candidates = [
        row
        for row in diagnostic_rows
        if row.get("diagnostic_label") in {"needs_manual_review", "exclude_from_current_eval", "diagnostic_active"}
    ][:80]
    manual_csv_rows = [
        {
            "sample_id": row.get("sample_id", ""),
            "query_type": row.get("query_type", ""),
            "query": row.get("query", ""),
            "target_doc_id": row.get("target_doc_id", ""),
            "stable_target_block_ids": "|".join(row.get("stable_target_block_ids") or []),
            "target_semantic_type": row.get("target_semantic_type", ""),
            "target_semantic_summary": row.get("target_semantic_summary", ""),
            "target_text_preview": row.get("target_text_preview", ""),
            "demotion_reason": row.get("demotion_reason", ""),
            "review_priority": "high" if row.get("diagnostic_label") == "needs_manual_review" else "medium",
        }
        for row in manual_candidates
    ]
    write_csv(
        OUT_DIR / "manual_review_pack_v2.csv",
        manual_csv_rows,
        [
            "sample_id",
            "query_type",
            "query",
            "target_doc_id",
            "stable_target_block_ids",
            "target_semantic_type",
            "target_semantic_summary",
            "target_text_preview",
            "demotion_reason",
            "review_priority",
        ],
    )
    write_text(
        OUT_DIR / "manual_review_pack_v2.md",
        "\n".join(
            [
                "# Manual Review Pack V2",
                "",
                f"- rows: {len(manual_csv_rows)}",
                "- priority: semantically useful but hard to naturalize, table_content boundary cases, caption fragments, and possible lexical-to-main promotions.",
                "",
                "See manual_review_pack_v2.csv for row-level details.",
            ]
        ),
    )

    over_semantic = sem_counts.most_common(1)[0] if sem_counts else ("none", 0)
    strains_primer = sum(
        count
        for sem, count in sem_counts.items()
        if sem in {"strains_plasmids", "primers_oligonucleotides"}
    )
    overdominance = over_semantic[1] / max(len(strict_rows), 1) > 0.35
    needs_further = overdominance or strains_primer / max(len(strict_rows), 1) > 0.35
    write_text(
        OUT_DIR / "distribution_audit.md",
        "\n".join(
            [
                "# Distribution Audit",
                "",
                "## Query Type Counts",
                *[f"- {key}: {value}" for key, value in sorted(qt_counts.items())],
                "",
                "## Target Semantic Type Counts",
                *[f"- {key}: {value}" for key, value in sem_counts.most_common()],
                "",
                "## Query Length Distribution",
                *[f"- {key}: {value}" for key, value in length_bins.items()],
                "",
                "## Risk Distributions",
                f"- strict hard-rule violations: {sum(strict_violation_counts.values())}",
                f"- strict copy-risk rows: {copy_risk}",
                f"- strict ID/number-density risk rows: {id_risk}",
                f"- duplicate/pattern audit rows: {len(duplicate_rows)}",
                f"- strains/plasmids/primers proportion: {strains_primer}/{len(strict_rows)}",
                "",
                "## Required Answers",
                f"- Semantic overdominance: {'yes' if overdominance else 'no'}; largest={over_semantic[0]} ({over_semantic[1]}).",
                f"- Further sampling demotion needed: {'yes' if needs_further else 'no'}.",
                f"- Strict main usable for Phase 5F-4: {'yes' if recommend_phase5f4 else 'no'}.",
            ]
        ),
    )

    write_text(
        OUT_DIR / "summary.md",
        "\n".join(
            [
                "# Phase 5F-3C Semantic Enhancement Summary",
                "",
                f"1. Input main count: {len(main_rows)}",
                f"2. strict_main_v2 count: {len(strict_rows)}",
                "3. Query type counts: "
                + ", ".join(f"{key}={value}" for key, value in sorted(qt_counts.items())),
                f"4. normal_control restored to 30: {str(normal_selected == 30).lower()} ({normal_selected})",
                f"5. Rewrite count: {rewrite_count}",
                f"6. Kept original query count: {kept_count}",
                f"7. Demoted to lexical_stress count: {final_counts.get('lexical_stress', 0)}",
                "8. Demoted to diagnostic/manual_review/exclude count: "
                f"{final_counts.get('diagnostic_active', 0) + final_counts.get('needs_manual_review', 0) + final_counts.get('exclude_from_current_eval', 0)}",
                "9. Current hard-rule violations: "
                + (
                    ", ".join(f"{key}={value}" for key, value in current_violation_counts.most_common())
                    if current_violation_counts
                    else "none"
                ),
                f"10. strict_main_v2 CAPTION residue: {strict_caption_residue}",
                f"11. strict_main_v2 target_chunk_id_only fields: {strict_target_chunk_id_only}",
                f"12. strict_main_v2 needs_manual_review rows: {strict_needs_manual}",
                f"13. strict_main_v2 OCR/image/row-cell structured table rows: {strict_ability_violations}",
                f"14. Stable target coverage: {stable_count}/{len(strict_rows)}",
                "15. Query length distribution: "
                + ", ".join(f"{key}={value}" for key, value in length_bins.items()),
                f"16. Copy target_text_preview risk: {copy_risk}",
                f"17. ID / number density risk: {id_risk}",
                f"18. Recommend entering Phase 5F-4: {'yes' if recommend_phase5f4 else 'no'}",
                "19. Phase 5F-4 should use strict_main_eval_set_v2.",
                "20. lexical_stress_eval_set_v2 should be reported separately.",
                "21. Retrieval changes needed: no.",
                "22. Rebuild index needed: no.",
                "23. Qwen/RAGAS needed: no.",
                "",
                f"Missing optional/expected inputs recorded: {len(missing_inputs)}",
                *[f"- {path}" for path in missing_inputs],
            ]
        ),
    )

    write_text(
        OUT_DIR / "before_after_comparison.md",
        "\n".join(
            [
                "# Before / After Comparison",
                "",
                f"- before strict_main_eval_set.jsonl: {len(main_rows)}",
                f"- after strict_main_eval_set_v2.jsonl: {len(strict_rows)}",
                f"- before normal_control: {Counter(row.get('query_type', '') for row in main_rows).get('normal_control', 0)}",
                f"- after normal_control: {normal_selected}",
                f"- after lexical_stress_eval_set_v2: {len(lexical_rows)}",
                f"- after diagnostic_demoted_from_main_v2: {len(diagnostic_rows)}",
                "",
                "The v2 strict set is intentionally smaller where targets or queries failed semantic naturalness or hard-rule lint.",
            ]
        ),
    )

    write_text(
        OUT_DIR / "next_phase_plan.md",
        """# Next Phase Plan

- Phase 5F-4 should use strict_main_eval_set_v2 as the strict main input.
- lexical_stress_eval_set_v2 should be run and reported separately.
- diagnostic_demoted_from_main_v2 and manual_review_pack_v2 do not enter main.
- Do not tune retrieval parameters for this phase.
- Do not call Qwen or RAGAS.
- Do not rebuild indexes; prefer existing indexes or a light sanity check only.
- If strict_main_v2 is considered too small for a formal benchmark, use it as a sanity set rather than expanding it with lower-quality samples.
""",
    )

    stats = {
        "input_main_count": len(main_rows),
        "strict_main_v2_count": len(strict_rows),
        "query_type_counts": dict(sorted(qt_counts.items())),
        "target_semantic_type_counts": dict(sem_counts.most_common()),
        "normal_control_count": normal_selected,
        "normal_control_restored_to_30": normal_selected == 30,
        "rewrite_count": rewrite_count,
        "kept_original_count": kept_count,
        "lexical_stress_demoted_from_main_count": final_counts.get("lexical_stress", 0),
        "diagnostic_manual_exclude_demoted_from_main_count": final_counts.get("diagnostic_active", 0)
        + final_counts.get("needs_manual_review", 0)
        + final_counts.get("exclude_from_current_eval", 0),
        "lexical_stress_v2_count": len(lexical_rows),
        "diagnostic_demoted_from_main_v2_count": len(diagnostic_rows),
        "current_hard_rule_violation_counts": dict(current_violation_counts.most_common()),
        "strict_hard_rule_violation_counts": dict(strict_violation_counts.most_common()),
        "strict_caption_residue": strict_caption_residue,
        "strict_target_chunk_id_only": strict_target_chunk_id_only,
        "strict_needs_manual_review": strict_needs_manual,
        "strict_ocr_image_or_row_cell": strict_ability_violations,
        "stable_target_coverage": {"covered": stable_count, "total": len(strict_rows)},
        "query_length_distribution": length_bins,
        "copy_target_preview_risk": copy_risk,
        "id_number_density_risk": id_risk,
        "recommend_phase5f4": recommend_phase5f4,
        "missing_inputs": missing_inputs,
        "distribution": distribution,
    }
    (OUT_DIR / "semantic_enhancement_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = read_jsonl(STRICT_MAIN_IN)
    lexical_input_rows = read_jsonl(LEXICAL_IN)
    normal_rows = read_jsonl(GOOD_NORMAL_IN)
    all_rows_for_chunks = main_rows + normal_rows
    chunks = load_chunk_texts(all_rows_for_chunks)
    missing_inputs = [str(path) for path in EXPECTED_INPUTS if not path.exists()]

    write_schema_docs()

    used_normal_queries: set[str] = set()
    audit_rows: list[dict[str, Any]] = []
    strict_candidates: list[dict[str, Any]] = []
    lexical_rows: list[dict[str, Any]] = [previous_lexical_row(row) for row in lexical_input_rows]
    diagnostic_rows: list[dict[str, Any]] = []
    source_by_sample = {row.get("sample_id", ""): row for row in main_rows}

    for row in main_rows:
        audit = evaluate_main_row(row, chunks, used_normal_queries)
        audit_rows.append(audit)
        if audit["final_label"] in {"main_strict", "main_rewritten"}:
            strict_candidates.append(strict_row_from_audit(row, audit))
        elif audit["final_label"] == "lexical_stress":
            lexical_rows.append(lexical_row_from_audit(row, audit))
        else:
            diagnostic_rows.append(diagnostic_row_from_audit(row, audit))

    _, normal_audit_rows, normal_strict_rows = process_normal_recovery(normal_rows, chunks)
    non_normal_candidates = [row for row in strict_candidates if row.get("query_type") != "normal_control"]
    combined_strict_candidates = non_normal_candidates + normal_strict_rows
    strict_rows, cap_demotions = dedupe_and_cap_strict(combined_strict_candidates)
    cap_demotions_by_id = {row.get("sample_id", ""): row for row in cap_demotions}
    for audit in audit_rows:
        demotion = cap_demotions_by_id.get(audit.get("sample_id", ""))
        if not demotion:
            continue
        audit["semantic_decision"] = "move_to_lexical_stress"
        audit["final_label"] = "lexical_stress"
        audit["should_be_main_strict"] = "false"
        audit["should_be_lexical_stress"] = "true"
        audit["rationale"] = issue_join(
            [
                audit.get("rationale", ""),
                f"distribution cap: {demotion.get('demotion_reason', '')}",
            ]
        )
    if STRICT_MAIN_V2_SAMPLE_ID_ALLOWLIST:
        preserved_strict_rows: list[dict[str, Any]] = []
        allowlist_demotions: list[dict[str, str]] = []
        for row in strict_rows:
            if row.get("sample_id", "") in STRICT_MAIN_V2_SAMPLE_ID_ALLOWLIST:
                preserved_strict_rows.append(row)
                continue
            allowlist_demotions.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "query": row.get("query", ""),
                    "query_type": row.get("query_type", ""),
                    "target_doc_id": row.get("target_doc_id", ""),
                    "stable_target_block_ids": "|".join(row.get("stable_target_block_ids") or []),
                    "target_text_preview": row.get("target_text_preview", ""),
                    "demotion_reason": "preserve existing strict_main_v2 sample set while updating reviewed table_content queries",
                }
            )
        strict_rows = preserved_strict_rows
        cap_demotions.extend(allowlist_demotions)
        allowlist_demotions_by_id = {row.get("sample_id", ""): row for row in allowlist_demotions}
        for audit in audit_rows:
            demotion = allowlist_demotions_by_id.get(audit.get("sample_id", ""))
            if not demotion:
                continue
            audit["semantic_decision"] = "move_to_lexical_stress"
            audit["final_label"] = "lexical_stress"
            audit["should_be_main_strict"] = "false"
            audit["should_be_lexical_stress"] = "true"
            audit["rationale"] = issue_join(
                [
                    audit.get("rationale", ""),
                    demotion.get("demotion_reason", ""),
                ]
            )
    strict_sample_ids = {row.get("sample_id", "") for row in strict_rows}

    for demotion in cap_demotions:
        source = source_by_sample.get(demotion["sample_id"], demotion)
        lexical_rows.append(
            {
                "sample_id": demotion["sample_id"],
                "original_sample_id": source.get("original_sample_id") or demotion["sample_id"],
                "query_type": demotion.get("query_type", ""),
                "query": demotion.get("query", ""),
                "original_query": source.get("original_query") or source.get("query") or demotion.get("query", ""),
                "previous_query": source.get("query") or demotion.get("query", ""),
                "include_in_main_denominator": False,
                "diagnostic_label": "lexical_stress",
                "target_doc_id": demotion.get("target_doc_id", ""),
                "stable_target_block_ids": demotion.get("stable_target_block_ids", "").split("|")
                if demotion.get("stable_target_block_ids")
                else [],
                "target_text_preview": demotion.get("target_text_preview", ""),
                "demotion_reason": demotion.get("demotion_reason", ""),
            }
        )

    deduped_lexical: list[dict[str, Any]] = []
    lexical_seen: set[str] = set()
    for row in lexical_rows:
        key = f"{row.get('sample_id','')}|{row.get('query','')}|{row_target_key(row)}"
        if row.get("sample_id") in strict_sample_ids:
            continue
        if key in lexical_seen:
            continue
        lexical_seen.add(key)
        row["include_in_main_denominator"] = False
        row["diagnostic_label"] = "lexical_stress"
        deduped_lexical.append(row)
    lexical_rows = deduped_lexical

    duplicate_rows = make_duplicate_audit(strict_rows, audit_rows)
    distribution = {
        "query_type_counts": dict(Counter(row.get("query_type", "") for row in strict_rows)),
        "target_semantic_type_counts": dict(Counter(row.get("target_semantic_type", "") for row in strict_rows)),
        "doc_counts": dict(Counter(row.get("target_doc_id", "") for row in strict_rows).most_common(20)),
    }

    write_csv(OUT_DIR / "semantic_enhancement_audit.csv", audit_rows, AUDIT_FIELDS)
    rewrite_rows = [
        {
            "sample_id": row["sample_id"],
            "query_type": row["query_type"],
            "original_query": row["original_query"],
            "current_query": row["current_query"],
            "rewritten_query": row["rewritten_query"],
            "target_doc_id": row["target_doc_id"],
            "stable_target_block_ids": row["stable_target_block_ids"],
            "target_semantic_summary": row["target_semantic_summary"],
            "user_intent": row["user_intent"],
            "rewrite_reason": row["rewrite_reason"],
            "hard_rule_passed": str(not row["final_hard_rule_violations"]).lower(),
            "rewrite_confidence": row["rewrite_confidence"],
            "risk_if_kept": row["current_hard_rule_violations"]
            or "semantic quality would remain caption-derived without rewrite",
        }
        for row in audit_rows
        if row["semantic_decision"] == "rewrite_and_keep_main"
    ]
    write_csv(OUT_DIR / "rewrite_ledger_v2.csv", rewrite_rows, REWRITE_FIELDS)
    write_jsonl(OUT_DIR / "strict_main_eval_set_v2.jsonl", strict_rows)
    write_csv(OUT_DIR / "normal_recovery_audit.csv", normal_audit_rows, NORMAL_RECOVERY_FIELDS)
    write_jsonl(OUT_DIR / "lexical_stress_eval_set_v2.jsonl", lexical_rows)
    write_jsonl(OUT_DIR / "diagnostic_demoted_from_main_v2.jsonl", diagnostic_rows)
    write_csv(OUT_DIR / "duplicate_and_pattern_audit.csv", duplicate_rows, DUP_FIELDS)

    write_markdown_reports(
        main_rows=main_rows,
        strict_rows=strict_rows,
        audit_rows=audit_rows,
        lexical_rows=lexical_rows,
        diagnostic_rows=diagnostic_rows,
        normal_audit=normal_audit_rows,
        duplicate_rows=duplicate_rows,
        missing_inputs=missing_inputs,
        distribution=distribution,
    )


if __name__ == "__main__":
    main()
