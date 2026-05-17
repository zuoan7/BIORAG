#!/usr/bin/env python3
"""Render Phase7G expanded table review pack artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extraction import build_expanded_table_review_candidates as builder
from scripts.extraction import export_table_review_crops


DEFAULT_OUTPUT_DIR = ROOT / "data/experiments/v7_phase7_expanded_table_review_pack"
DEFAULT_REPORT_DIR = ROOT / "reports/v7_phase7_expanded_table_review_pack"
DEFAULT_PDF_DIR = ROOT / "data/paper_round1/paper"

REVIEW_DECISION_VALUES = [
    "accept_confirmed_seed_candidate",
    "accept_partial_seed_candidate",
    "reject_boundary",
    "reject_grid",
    "needs_rule_fix",
    "backlog",
    "skip",
]
CORE_OK_VALUES = ["yes", "no", "unclear"]
OPTIONAL_OK_VALUES = ["yes", "warning", "no", "not_applicable", "unchecked"]

REVIEW_LABEL_FIELDS = [
    "candidate_id",
    "review_decision",
    "boundary_ok",
    "grid_ok",
    "key_values_ok",
    "unit_or_note_ok",
    "reference_ok",
    "review_notes",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "review_priority",
    "suggested_decision",
    "risk_tags",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
]

REVIEW_INDEX_FIELDS = [
    "candidate_id",
    "table_object_id",
    "doc_id",
    "table_id",
    "caption",
    "page",
    "routing_status",
    "auto_score",
    "review_priority",
    "suggested_decision",
    "risk_tags",
    "table_type_tags",
    "markdown_path",
    "csv_path",
    "pdf_crop_path",
    "crop_status",
    "source_span_granularity",
    "value_bboxes_available",
    "cell_bboxes_available",
    "warnings_summary",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def semicolon(values: list[Any] | tuple[Any, ...] | set[Any]) -> str:
    cleaned = [normalize_space(value) for value in values if normalize_space(value)]
    return ";".join(dict.fromkeys(cleaned)) if cleaned else "none"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_candidates(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "candidate_pool_raw.jsonl"
    if not path.exists():
        raise SystemExit(f"candidate pool missing: {rel(path)}")
    return load_jsonl(path)


def sort_key(candidate: dict[str, Any]) -> tuple[int, float, str]:
    return (
        builder.REVIEW_PRIORITY_ORDER.get(candidate.get("review_priority", "auto_excluded"), 9),
        -float(candidate.get("auto_score") or 0.0),
        candidate.get("candidate_id", ""),
    )


def select_review_candidates(candidates: list[dict[str, Any]], review_target: int) -> list[dict[str, Any]]:
    allowed = [row for row in candidates if row.get("review_priority") != "auto_excluded"]
    allowed.sort(key=sort_key)
    return allowed[:review_target]


def md_escape(value: Any) -> str:
    return normalize_space(value).replace("|", "\\|")


def rows_for_candidate(candidate: dict[str, Any]) -> list[list[str]]:
    rows = candidate.get("rows") or []
    normalized = []
    for row in rows:
        if isinstance(row, list):
            normalized.append([normalize_space(cell) for cell in row])
    return normalized


def write_candidate_csv(candidate: dict[str, Any], path: Path) -> None:
    rows = rows_for_candidate(candidate)
    max_cols = max((len(row) for row in rows), default=0)
    fieldnames = ["row_index"] + [f"col_{index:03d}" for index in range(1, max_cols + 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(rows, start=1):
            writer.writerow(
                {"row_index": row_index}
                | {f"col_{index:03d}": row[index - 1] if index - 1 < len(row) else "" for index in range(1, max_cols + 1)}
            )


def markdown_table(rows: list[list[str]], max_rows: int = 12, max_cols: int = 8) -> list[str]:
    if not rows:
        return ["_无可用 extracted rows；请优先查看 PDF crop 或跳过。_"]
    cols = min(max((len(row) for row in rows), default=0), max_cols)
    header = [f"col_{index:03d}" for index in range(1, cols + 1)]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * cols) + " |"]
    for row in rows[:max_rows]:
        values = [md_escape(row[index]) if index < len(row) else "" for index in range(cols)]
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_仅预览前 {max_rows} 行，完整内容见 CSV。_")
    return lines


def write_markdown_card(candidate: dict[str, Any], path: Path) -> str:
    rows = rows_for_candidate(candidate)
    warning_summary = semicolon((candidate.get("warnings") or [])[:12])
    lines = [
        f"# {candidate['candidate_id']}",
        "",
        f"- candidate_id：`{candidate.get('candidate_id', '')}`",
        f"- table_object_id：`{candidate.get('table_object_id', '')}`",
        f"- doc_id：`{candidate.get('doc_id', '')}`",
        f"- table_id：`{candidate.get('table_id', '')}`",
        f"- caption：{md_escape(candidate.get('caption', ''))}",
        f"- page：`{candidate.get('page', '')}`",
        f"- routing_status：`{candidate.get('routing_status', '')}`",
        f"- auto_score：`{candidate.get('auto_score', '')}`",
        f"- review_priority：`{candidate.get('review_priority', '')}`",
        f"- suggested_decision：`{candidate.get('suggested_decision', '')}`",
        f"- risk_tags：`{semicolon(candidate.get('risk_tags') or [])}`",
        f"- PDF crop path：`{candidate.get('pdf_crop_path', '')}`",
        f"- CSV path：`{candidate.get('csv_path', '')}`",
        f"- source_span_granularity：`{candidate.get('source_span_granularity', '')}`",
        f"- value_bboxes_available：`{str(candidate.get('value_bboxes_available', False)).lower()}`",
        f"- cell_bboxes_available：`{str(candidate.get('cell_bboxes_available', False)).lower()}`",
        f"- warnings summary：`{warning_summary}`",
        "",
        "## Extracted Table Preview",
        "",
        *markdown_table(rows),
        "",
        "## Review Checklist",
        "",
        "- 是否是目标表",
        "- 表格边界是否正确",
        "- 行列是否基本正确",
        "- 关键值是否一致",
        "- 是否可接受为 confirmed / partial / reject",
    ]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def column_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def inline_cell(ref: str, value: Any) -> str:
    text = escape(normalize_space(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_minimal_xlsx(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    header_cells = [inline_cell(f"{column_letter(index)}1", field) for index, field in enumerate(fieldnames, start=1)]
    sheet_rows.append(f'<row r="1">{"".join(header_cells)}</row>')
    for row_index, row in enumerate(rows, start=2):
        cells = [
            inline_cell(f"{column_letter(col_index)}{row_index}", row.get(field, ""))
            for col_index, field in enumerate(fieldnames, start=1)
        ]
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    def validation(col_name: str, values: list[str]) -> str:
        col_index = fieldnames.index(col_name) + 1
        col = column_letter(col_index)
        formula = escape('"' + ",".join(values) + '"')
        return f'<dataValidation type="list" allowBlank="1" sqref="{col}2:{col}1048576"><formula1>{formula}</formula1></dataValidation>'

    validations = [
        validation("review_decision", REVIEW_DECISION_VALUES),
        validation("boundary_ok", CORE_OK_VALUES),
        validation("grid_ok", CORE_OK_VALUES),
        validation("key_values_ok", CORE_OK_VALUES),
        validation("unit_or_note_ok", OPTIONAL_OK_VALUES),
        validation("reference_ok", OPTIONAL_OK_VALUES),
    ]
    dimension = f"A1:{column_letter(len(fieldnames))}{max(len(rows) + 1, 1)}"
    worksheet = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dimension}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft"/></sheetView></sheetViews>
  <sheetData>{''.join(sheet_rows)}</sheetData>
  <dataValidations count="{len(validations)}">{''.join(validations)}</dataValidations>
</worksheet>'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="review" sheetId="1" r:id="rId1"/></sheets></workbook>'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>'''
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)


def review_index_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "table_object_id": candidate.get("table_object_id", ""),
        "doc_id": candidate.get("doc_id", ""),
        "table_id": candidate.get("table_id", ""),
        "caption": normalize_space(candidate.get("caption", ""))[:500],
        "page": candidate.get("page", ""),
        "routing_status": candidate.get("routing_status", ""),
        "auto_score": candidate.get("auto_score", ""),
        "review_priority": candidate.get("review_priority", ""),
        "suggested_decision": candidate.get("suggested_decision", ""),
        "risk_tags": semicolon(candidate.get("risk_tags") or []),
        "table_type_tags": semicolon(candidate.get("table_type_tags") or []),
        "markdown_path": candidate.get("markdown_path", ""),
        "csv_path": candidate.get("csv_path", ""),
        "pdf_crop_path": candidate.get("pdf_crop_path", ""),
        "crop_status": candidate.get("crop_status", ""),
        "source_span_granularity": candidate.get("source_span_granularity", ""),
        "value_bboxes_available": str(bool(candidate.get("value_bboxes_available"))).lower(),
        "cell_bboxes_available": str(bool(candidate.get("cell_bboxes_available"))).lower(),
        "warnings_summary": semicolon((candidate.get("warnings") or [])[:12]),
    }


def label_template_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "review_decision": "",
        "boundary_ok": "",
        "grid_ok": "",
        "key_values_ok": "",
        "unit_or_note_ok": "unchecked",
        "reference_ok": "unchecked",
        "review_notes": "",
        "table_object_id": candidate.get("table_object_id", ""),
        "doc_id": candidate.get("doc_id", ""),
        "table_id": candidate.get("table_id", ""),
        "caption": normalize_space(candidate.get("caption", ""))[:500],
        "page": candidate.get("page", ""),
        "review_priority": candidate.get("review_priority", ""),
        "suggested_decision": candidate.get("suggested_decision", ""),
        "risk_tags": semicolon(candidate.get("risk_tags") or []),
        "markdown_path": candidate.get("markdown_path", ""),
        "csv_path": candidate.get("csv_path", ""),
        "pdf_crop_path": candidate.get("pdf_crop_path", ""),
        "crop_status": candidate.get("crop_status", ""),
    }


def update_scored_csv(candidates: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [builder.scored_csv_row(row) for row in sorted(candidates, key=sort_key)]
    write_csv(rows, output_dir / "candidate_pool_scored.csv", builder.SCORED_FIELDS)


def md_counter(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- 无"]
    return [f"- `{key}`：{value}" for key, value in counter.most_common()]


def counter_from_candidates(candidates: list[dict[str, Any]], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for candidate in candidates:
        value = candidate.get(key)
        if isinstance(value, list):
            for item in value:
                counter[normalize_space(item) or "empty"] += 1
        else:
            counter[normalize_space(value) or "empty"] += 1
    return counter


def write_human_review_instructions(report_dir: Path) -> None:
    lines = [
        "# Human Review Instructions",
        "",
        "## 快速核验流程",
        "",
        "每张表只看三样：",
        "",
        "1. PDF crop：判断是不是目标表、边界有没有截断或混入；",
        "2. Markdown preview：判断整体行列是否基本对；",
        "3. CSV：判断关键值和 literal 是否一致。",
        "",
        "## 最少填写字段",
        "",
        "用户只需填写：",
        "",
        "- `review_decision`；",
        "- `boundary_ok`；",
        "- `grid_ok`；",
        "- `key_values_ok`。",
        "",
        "`unit_or_note_ok` / `reference_ok` 可以只在明显有问题时填写。",
        "",
        "## 决策规则",
        "",
        "`accept_confirmed_seed_candidate`：是目标表；边界正确；行列基本正确；关键值一致；unit / footnote / reference 没有明显阻断，或只是 warning。",
        "",
        "`accept_partial_seed_candidate`：是目标表；边界基本正确；表格大体可用；但 unit / footnote / reference / split cell 有不确定。",
        "",
        "`reject_boundary`：不是目标表；截断严重；混入正文、figure 或其他表。",
        "",
        "`reject_grid`：目标表可能对，但行列错位严重；CSV/Markdown 不可用。",
        "",
        "`needs_rule_fix`：表格有价值；但 split/merged/row continuation 等规则需要修。",
        "",
        "`backlog`：当前无法判断；PDF text layer / crop / layout 证据不足。",
        "",
        "`skip`：暂不审。",
        "",
        "## 推荐人工工作量",
        "",
        "优先只审 `P0_quick_review`。`P1_review` 只在 P0 数量不足时审。`P2_optional_spotcheck` 可跳过。",
        "",
        "不要逐 cell 标注。不要尝试修表。不要补 gold。只做整体可用性标签。",
    ]
    (report_dir / "human_review_instructions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_generation_report(
    report_dir: Path,
    review_candidates: list[dict[str, Any]],
    markdown_count: int,
    csv_count: int,
    crop_ok: int,
    crop_failed: int,
    output_dir: Path,
) -> None:
    lines = [
        "# Review Pack Generation Report",
        "",
        f"1. Markdown card 数量：{markdown_count}",
        f"2. CSV table 数量：{csv_count}",
        f"3. PDF crop 数量：{crop_ok}",
        f"4. crop failed 数量：{crop_failed}",
        f"5. review_sheet.xlsx 是否生成：{'是' if (output_dir / 'review_sheet.xlsx').exists() else '否'}",
        f"6. review_labels_template.csv 是否生成：{'是' if (output_dir / 'review_labels_template.csv').exists() else '否'}",
        "7. 人工核验如何执行：按 `human_review_instructions.md`，优先打开 crop、Markdown、CSV，只填写 4 个核心字段。",
        "8. 后续如何导入人工 label：后续阶段应只读取用户填写后的 `review_labels_template.csv`，再按 review_decision 转为 confirmed_seed / partial_seed / reject；本轮不做转换。",
        "",
        "## review_priority 分布",
        *md_counter(counter_from_candidates(review_candidates, "review_priority")),
        "",
        "## suggested_decision 分布",
        *md_counter(counter_from_candidates(review_candidates, "suggested_decision")),
    ]
    (report_dir / "review_pack_generation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    report_dir: Path,
    output_dir: Path,
    all_candidates: list[dict[str, Any]],
    review_candidates: list[dict[str, Any]],
    markdown_count: int,
    csv_count: int,
    crop_ok: int,
    crop_failed: int,
) -> None:
    auto_excluded = [row for row in all_candidates if row.get("review_priority") == "auto_excluded"]
    lines = [
        "# Phase7G Summary",
        "",
        "## 1. 本轮生成文件",
        f"- `{rel(output_dir / 'candidate_pool_raw.jsonl')}`",
        f"- `{rel(output_dir / 'candidate_pool_scored.csv')}`",
        f"- `{rel(output_dir / 'review_pack_index.csv')}`",
        f"- `{rel(output_dir / 'review_labels_template.csv')}`",
        f"- `{rel(output_dir / 'review_sheet.xlsx')}`",
        f"- `{rel(output_dir / 'auto_excluded_candidates.csv')}`",
        f"- `{rel(output_dir / 'markdown_cards/all_review_cards.md')}`",
        f"- `{rel(output_dir / 'csv_tables')}/`",
        f"- `{rel(output_dir / 'pdf_crops')}/`",
        f"- `{rel(report_dir / 'phase7g_guardrail.md')}`",
        f"- `{rel(report_dir / 'candidate_pool_construction_report.md')}`",
        f"- `{rel(report_dir / 'review_pack_generation_report.md')}`",
        f"- `{rel(report_dir / 'human_review_instructions.md')}`",
        f"- `{rel(report_dir / 'phase7g_summary.md')}`",
        "",
        "## 2. 新增 / 修改脚本",
        "- 新增：`scripts/extraction/build_expanded_table_review_candidates.py`",
        "- 新增：`scripts/extraction/render_expanded_table_review_pack.py`",
        "- 新增：`scripts/extraction/export_table_review_crops.py`",
        "",
        "## 3. 新增测试",
        "- 新增：`tests/test_phase7_expanded_review_pack.py`",
        "",
        "## 4. 是否扩大候选池",
        "- 是，本轮扩大为离线 review candidate pool。",
        "",
        f"## 5. raw candidate pool 数量：{len(all_candidates)}",
        f"## 6. review pack candidate 数量：{len(review_candidates)}",
        "",
        "## 7. review_priority 统计",
        *md_counter(counter_from_candidates(all_candidates, "review_priority")),
        "",
        "## 8. suggested_decision 统计",
        *md_counter(counter_from_candidates(all_candidates, "suggested_decision")),
        "",
        f"## 9. auto_excluded 数量：{len(auto_excluded)}",
        f"## 10. Markdown / CSV / crop 生成数量：Markdown={markdown_count}，CSV={csv_count}，crop_ok={crop_ok}，crop_failed={crop_failed}",
        "",
        "## 11. review sheet 是否方便人工填写",
        "- 是。主 sheet 冻结首行，并为 review_decision、boundary_ok、grid_ok、key_values_ok、unit_or_note_ok、reference_ok 设置下拉选项；P0 排在最前。",
        "",
        "## 12. 是否构造 gold seed",
        "- 否。本轮不构造 confirmed_seed 或 partial_seed。",
        "",
        "## 13. 是否建议用户开始人工 review",
        "- 是。建议先审 `P0_quick_review`。",
        "",
        "## 14. 是否建议进入 production",
        "- 否。本轮结果不是 production readiness。",
        "",
        "## 15. baseline / guardrail 是否漂移",
        "- 未发现漂移；未修改 official baseline、configs、baseline registry、chunks、BM25 或 Milvus。",
        "",
        "## 16. Route C 是否仍只是 backlog",
        "- 是，Route C 仍只是 backlog。",
        "",
        "## 17. 明确未执行事项",
        "- 未构造 confirmed gold。",
        "- 未构造 partial seed。",
        "- 未运行 validation / coverage / flat comparison。",
        "- 未读取或查询 BM25 index。",
        "- 未访问或写入 Milvus。",
        "- 未运行 retrieval、embedding、rerank、Qwen、RAGAS、OCR、VLM。",
        "- 未修改 ingestion 主链路、production pipeline、official dataset、official baseline、configs 或 README。",
        "- 未进入 Route C implementation。",
    ]
    (report_dir / "phase7g_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_review_pack(
    output_dir: Path,
    report_dir: Path,
    pdf_dir: Path,
    review_target: int,
    export_crops_enabled: bool = True,
) -> dict[str, Any]:
    candidates = read_candidates(output_dir)
    review_candidates = select_review_candidates(candidates, review_target)
    review_ids = {row["candidate_id"] for row in review_candidates}

    markdown_dir = output_dir / "markdown_cards"
    csv_dir = output_dir / "csv_tables"
    crop_dir = output_dir / "pdf_crops"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    crop_results: dict[str, dict[str, str]] = {}
    if export_crops_enabled:
        crop_results = export_table_review_crops.export_crops(review_candidates, crop_dir, pdf_dir)

    markdown_texts = []
    markdown_count = 0
    csv_count = 0
    for candidate in review_candidates:
        cid = candidate["candidate_id"]
        candidate["markdown_path"] = rel(markdown_dir / f"{cid}.md")
        candidate["csv_path"] = rel(csv_dir / f"{cid}.csv")
        crop_result = crop_results.get(cid, {})
        if crop_result:
            candidate["crop_status"] = crop_result.get("crop_status", "failed")
            candidate["pdf_crop_path"] = crop_result.get("pdf_crop_path", candidate.get("pdf_crop_path", ""))
            candidate["crop_error"] = crop_result.get("crop_error", "")
        elif not export_crops_enabled:
            candidate["crop_status"] = "not_exported_in_test"
            candidate["pdf_crop_path"] = rel(crop_dir / f"{cid}.png")
        write_candidate_csv(candidate, csv_dir / f"{cid}.csv")
        csv_count += 1
        markdown_texts.append(write_markdown_card(candidate, markdown_dir / f"{cid}.md"))
        markdown_count += 1

    (markdown_dir / "all_review_cards.md").write_text("\n\n---\n\n".join(markdown_texts) + "\n", encoding="utf-8")

    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    for candidate in review_candidates:
        candidate_by_id[candidate["candidate_id"]] = candidate
    for candidate in candidates:
        if candidate["candidate_id"] not in review_ids and candidate.get("review_priority") != "auto_excluded":
            candidate["crop_status"] = "not_in_review_pack"

    index_rows = [review_index_row(row) for row in review_candidates]
    label_rows = [label_template_row(row) for row in review_candidates]
    write_csv(index_rows, output_dir / "review_pack_index.csv", REVIEW_INDEX_FIELDS)
    write_csv(label_rows, output_dir / "review_labels_template.csv", REVIEW_LABEL_FIELDS)
    write_minimal_xlsx(label_rows, output_dir / "review_sheet.xlsx", REVIEW_LABEL_FIELDS)
    update_scored_csv(list(candidate_by_id.values()), output_dir)

    crop_ok = sum(1 for row in review_candidates if row.get("crop_status") == "ok")
    crop_failed = sum(1 for row in review_candidates if row.get("crop_status") == "failed")
    write_human_review_instructions(report_dir)
    write_generation_report(report_dir, review_candidates, markdown_count, csv_count, crop_ok, crop_failed, output_dir)
    write_summary(report_dir, output_dir, list(candidate_by_id.values()), review_candidates, markdown_count, csv_count, crop_ok, crop_failed)

    return {
        "review_candidates": len(review_candidates),
        "markdown_cards": markdown_count,
        "csv_tables": csv_count,
        "crop_ok": crop_ok,
        "crop_failed": crop_failed,
        "review_sheet": rel(output_dir / "review_sheet.xlsx"),
        "review_labels_template": rel(output_dir / "review_labels_template.csv"),
    }


def run(args: argparse.Namespace) -> None:
    result = render_review_pack(args.output_dir, args.report_dir, args.pdf_dir, args.review_target, not args.skip_crops)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase7G expanded table review pack.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--review-target", type=int, default=40)
    parser.add_argument("--skip-crops", action="store_true")
    args = parser.parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.report_dir = resolve_path(args.report_dir)
    args.pdf_dir = resolve_path(args.pdf_dir)
    return args


if __name__ == "__main__":
    run(parse_args())
