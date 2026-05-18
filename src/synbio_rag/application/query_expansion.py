"""BioLexical BM25 query expansion for scientific/biomedical cross-language matching.

This is a prototype (Phase 14C), not a runtime main-chain module. It extracts
Latin/English biomedical terms from Chinese queries and expands them via a small
synonym map for BM25 sparse retrieval. It does NOT replace the original query,
only appends English/bio terms.
"""
from __future__ import annotations

import re
import unicodedata

# ── Unicode normalization ────────────────────────────────────────────

_PRIME_VARIANTS = str.maketrans({"\u2032": "'", "\u2018": "'", "\u2019": "'", "\u02bc": "'"})
_HYPHEN_VARIANTS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-"})
_FULLWIDTH_LATIN = str.maketrans({chr(i): chr(i - 0xFF00 + 0x20) for i in range(0xFF01, 0xFF5F)})


def normalize_text(text: str) -> str:
    """Unicode-normalize: prime variants, hyphens, fullwidth Latin."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PRIME_VARIANTS)
    text = text.translate(_HYPHEN_VARIANTS)
    text = text.translate(_FULLWIDTH_LATIN)
    return text


# ── Token extraction patterns (for query expansion, NOT for BM25 indexing) ──

_LATIN_TERM_RE = re.compile(r"[A-Za-z0-9'αβγμ\-+/_]+")
_CN_WORD_RE = re.compile(r"[\u4e00-\u9fff]{2,}")

# ── Synonym map ──────────────────────────────────────────────────────

SYNONYM_MAP: dict[str, list[str]] = {
    # HMO / sugars
    "hmo": ["hmo", "human milk oligosaccharide", "human milk oligosaccharides"],
    "2'-fl": ["2'-fl", "2'-fucosyllactose", "2-fucosyllactose", "2 fucosyllactose"],
    "3-fl": ["3-fl", "3-fucosyllactose", "3 fucosyllactose"],
    "6'-sl": ["6'-sl", "6'-sialyllactose", "6-sialyllactose"],
    "3'-sl": ["3'-sl", "3'-sialyllactose", "3-sialyllactose"],
    # Neu5Ac / sialic acid
    "neu5ac": ["neu5ac", "neuac", "n-acetylneuraminic acid", "sialic acid"],
    "sialic acid": ["sialic acid", "neu5ac", "neuac", "n-acetylneuraminic acid"],
    # Pichia / hosts
    "pichia pastoris": ["pichia pastoris", "komagataella phaffii", "k. phaffii", "p. pastoris"],
    "pichia": ["pichia pastoris", "komagataella phaffii", "pichia"],
    "e. coli": ["e. coli", "escherichia coli"],
    "escherichia coli": ["escherichia coli", "e. coli"],
    # alpha mating factor
    "α-mating factor": ["α-mating factor", "alpha mating factor", "mfα", "mf-alpha"],
    "mfα": ["mfα", "mf-alpha", "α-mating factor", "alpha mating factor"],
    # glycosylation
    "glycosylation": ["glycosylation", "n-glycosylation", "o-glycosylation"],
    "n-glycosylation": ["n-glycosylation", "glycosylation"],
    # phosphorylation
    "phosphorylation": ["phosphorylation"],
    # secretion
    "secretion": ["secretion", "secretory", "secreted"],
    # osteoporosis / macrophage
    "osteoporosis": ["osteoporosis", "bone loss"],
    "macrophage": ["macrophage", "m1 macrophage", "macrophage polarization"],
    "m1": ["m1 macrophage", "macrophage polarization"],
    # FAM20
    "fam20a": ["fam20a", "fam20c"],
    "fam20c": ["fam20c", "fam20a"],
    # Kex2
    "kex2": ["kex2", "kex2 cleavage", "kex2 processing"],
    # CRISPR
    "crispr": ["crispr", "crispr-cas9", "crispr cas9"],
    "paste": ["paste", "crispr-directed integrase", "drag-and-drop genome insertion"],
    # PASTE
    "crispr-tmsd": ["crispr-tmsd", "crispr tmsd"],
    # SemiSWEET
    "semisweet": ["semisweet", "semi sweet", "semisweets"],
    # riboswitch
    "riboswitch": ["riboswitch", "riboswitches"],
    # Nan cluster
    "nan cluster": ["nan cluster", "nan gene cluster", "sialoregulon"],
    "nanr": ["nanr", "sialoregulon"],
    # promoter
    "promoter": ["promoter", "bidirectional promoter"],
    # OPN / osteopontin
    "osteopontin": ["osteopontin", "opn"],
    "opn": ["opn", "osteopontin"],
    # HAC1
    "hac1": ["hac1", "hac1 gene", "hac1p"],
}

# ── Chinese term dictionary (longest-match for CN word segmentation) ──

_CN_TERMS = sorted([
    "骨质疏松", "巨噬细胞", "巨噬细胞极化", "糖基化", "磷酸化", "分泌", "分泌表达",
    "启动子", "转运蛋白", "信号肽", "毕赤酵母", "大肠杆菌", "人乳寡糖", "唾液酸",
    "岩藻糖基乳糖", "岩藻糖", "半乳糖", "葡萄糖", "木糖", "阿拉伯糖",
    "发酵", "代谢工程", "代谢途径", "合成途径", "从头合成",
    "表达盒", "表达载体", "基因表达", "蛋白表达", "过量表达",
    "降解", "木质素", "生物修复", "生物制造", "底盘工程",
    "质粒", "染色体整合", "基因敲除", "基因缺失",
    "酶活", "催化效率", "底物特异性", "最适温度", "最适pH",
    "多酶级联", "共培养", "转录组", "蛋白组", "分泌组",
    "双向启动子", "多顺反子", "信号传导", "细胞表面展示",
    "产物抑制", "反馈抑制", "前体供给", "辅因子再生",
], key=len, reverse=True)  # sort by length descending for longest match

_CN_TERM_RE = re.compile("|".join(re.escape(t) for t in _CN_TERMS))


def extract_cn_terms(text: str) -> list[str]:
    """Extract Chinese biomedical terms via longest-match dictionary lookup."""
    found = []
    pos = 0
    while pos < len(text):
        m = _CN_TERM_RE.match(text, pos)
        if m:
            found.append(m.group())
            pos = m.end()
        else:
            pos += 1
    return list(set(found))


# ── Main expander ────────────────────────────────────────────────────

class BioLexicalQueryExpander:
    """Lightweight biomedical query expansion for BM25 sparse retrieval."""

    def __init__(self, max_terms: int = 32):
        self.max_terms = max_terms

    def expand(self, query: str) -> dict:
        raw = query
        norm = normalize_text(raw)

        # Extract Latin/English biomedical terms
        latin_terms = []
        for m in _LATIN_TERM_RE.finditer(norm):
            t = m.group().strip("'-/")
            if t and len(t) >= 2:
                latin_terms.append(t.lower())

        # Extract Chinese terms via dictionary
        cn_terms = extract_cn_terms(norm)

        # Build expanded terms via synonym map
        expanded = set()
        for t in latin_terms:
            expanded.add(t)
            aliases = SYNONYM_MAP.get(t, [])
            for a in aliases:
                expanded.add(a)

        # Add CN terms (unexpanded for now)
        for t in cn_terms:
            expanded.add(t)

        # Build BM25 query: raw CN + all expanded terms (deduped)
        terms_list = list(expanded)
        if len(terms_list) > self.max_terms:
            terms_list = terms_list[:self.max_terms]

        bm25_query = f"{raw} {' '.join(terms_list)}"

        return {
            "raw_query": raw,
            "normalized_query": norm,
            "latin_terms": latin_terms,
            "chinese_terms": cn_terms,
            "expanded_terms": terms_list,
            "bm25_expanded_query": bm25_query,
            "expansion_term_count": len(terms_list),
            "expansion_truncated": len(expanded) > self.max_terms,
        }
