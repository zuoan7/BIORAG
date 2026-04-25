import json
import re
from pathlib import Path

import fitz

PDF_DIR = Path("data/paper_round1/paper")
PARSED_DIR = Path("data/paper_round1/parsed")
OUT = Path("reports/pdf_extraction_audit_safe.json")

def safe_text(x):
    if x is None:
        return ""
    return str(x).replace("\n", " ").strip()

rows = []

for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    doc_id = pdf_path.stem
    parsed_path = PARSED_DIR / f"{doc_id}.json"

    row = {
        "pdf": pdf_path.name,
        "annot_count": 0,
        "annot_pages": [],
        "annot_samples": [],
        "image_count": 0,
        "image_pages": [],
        "parsed_exists": parsed_path.exists(),
        "parsed_chars": 0,
        "parsed_has_figure": False,
        "parsed_has_table": False,
        "parsed_has_supplementary": False,
    }

    try:
        doc = fitz.open(str(pdf_path))

        for page_idx, page in enumerate(doc, start=1):
            annots = list(page.annots() or [])
            if annots:
                row["annot_pages"].append(page_idx)

            for annot in annots:
                row["annot_count"] += 1
                info = getattr(annot, "info", {}) or {}
                content = safe_text(info.get("content"))
                title = safe_text(info.get("title"))
                subject = safe_text(info.get("subject"))
                sample = " | ".join(x for x in [title, subject, content] if x)
                if sample and len(row["annot_samples"]) < 5:
                    row["annot_samples"].append(sample[:300])

            imgs = page.get_images(full=True)
            if imgs:
                row["image_pages"].append(page_idx)
                row["image_count"] += len(imgs)

        doc.close()

        if parsed_path.exists():
            try:
                data = json.loads(parsed_path.read_text(encoding="utf-8"))
                parsed_text = json.dumps(data, ensure_ascii=False)

                row["parsed_chars"] = len(parsed_text)
                row["parsed_has_figure"] = bool(re.search(r"\b(fig\.|figure)\b", parsed_text, re.I))
                row["parsed_has_table"] = bool(re.search(r"\btable\b", parsed_text, re.I))
                row["parsed_has_supplementary"] = bool(re.search(r"\bsupplementary\b", parsed_text, re.I))
            except Exception as e:
                row["parsed_error"] = str(e)

    except Exception as e:
        row["error"] = str(e)

    rows.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"wrote {OUT}")
print("pdf_count:", len(rows))
print("with_annotations:", sum(1 for r in rows if r.get("annot_count", 0) > 0))
print("with_images:", sum(1 for r in rows if r.get("image_count", 0) > 0))
print("with_errors:", sum(1 for r in rows if r.get("error")))

print("\nPDFs with annotations:")
for r in rows:
    if r.get("annot_count", 0) > 0:
        print(r["pdf"], "annots=", r["annot_count"], "pages=", r.get("annot_pages"), "samples=", r.get("annot_samples"))

print("\nPDFs with images:")
for r in rows:
    if r.get("image_count", 0) > 0:
        print(r["pdf"], "images=", r["image_count"], "pages=", r.get("image_pages")[:10])
