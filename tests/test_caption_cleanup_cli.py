from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/ingestion/cleanup_false_fragment_captions.py"


def write_doc(input_dir: Path) -> None:
    doc = {
        "doc_id": "doc_cli",
        "source_file": "doc_cli.pdf",
        "parser_stage": "parsed_clean_v1",
        "pages": [
            {
                "page": 1,
                "text": "All strains are listed in\nTable 1. E.\ncoli DH5α was used.",
                "blocks": [
                    {
                        "block_id": "p1_b0001",
                        "type": "paragraph",
                        "text": "All strains are listed in",
                        "page": 1,
                        "section_path": ["Methods"],
                    },
                    {
                        "block_id": "p1_b0002",
                        "type": "table_caption",
                        "text": "Table 1. E.",
                        "page": 1,
                        "section_path": ["Methods"],
                    },
                    {
                        "block_id": "p1_b0003",
                        "type": "paragraph",
                        "text": "coli DH5α was used.",
                        "page": 1,
                        "section_path": ["Methods"],
                    },
                ],
            }
        ],
    }
    input_dir.mkdir(parents=True)
    (input_dir / "doc_cli.json").write_text(json.dumps(doc), encoding="utf-8")


def write_signoff(path: Path) -> None:
    fields = [
        "doc_id",
        "block_id",
        "block_type",
        "caption_text",
        "candidate_rule",
        "confidence",
        "label",
        "rationale",
        "recommended_cleanup_action",
        "risk_if_wrong",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "doc_id": "doc_cli",
                "block_id": "p1_b0002",
                "block_type": "table_caption",
                "caption_text": "Table 1. E.",
                "candidate_rule": "broken_organism_or_abbreviation_prefix",
                "confidence": "high",
                "label": "safe_to_demote",
                "rationale": "test",
                "recommended_cleanup_action": "demote",
                "risk_if_wrong": "test",
            }
        )


def write_protected(path: Path) -> None:
    fields = ["doc_id", "block_id", "caption_text", "protect_reason", "review_label"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()


def run_cli(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    audit_dir = tmp_path / "audit"
    signoff = tmp_path / "signoff.csv"
    protected = tmp_path / "protected.csv"
    write_doc(input_dir)
    write_signoff(signoff)
    write_protected(protected)
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--signoff_csv",
        str(signoff),
        "--protected_csv",
        str(protected),
        "--audit_dir",
        str(audit_dir),
        *extra,
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def test_dry_run_writes_audit_but_not_parsed_json(tmp_path: Path) -> None:
    result = run_cli(tmp_path, "--dry_run")

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "audit/cleanup_audit.csv").exists()
    assert (tmp_path / "audit/summary.md").exists()
    assert not (tmp_path / "output/doc_cli.json").exists()


def test_output_dir_must_be_isolated_from_input_dir(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    audit_dir = tmp_path / "audit"
    signoff = tmp_path / "signoff.csv"
    protected = tmp_path / "protected.csv"
    write_doc(input_dir)
    write_signoff(signoff)
    write_protected(protected)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(input_dir),
        "--signoff_csv",
        str(signoff),
        "--protected_csv",
        str(protected),
        "--audit_dir",
        str(audit_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 2
    assert "output_dir must not equal input_dir" in result.stderr


def test_rejects_production_parsed_clean_output_dir(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    audit_dir = tmp_path / "audit"
    signoff = tmp_path / "signoff.csv"
    protected = tmp_path / "protected.csv"
    write_doc(input_dir)
    write_signoff(signoff)
    write_protected(protected)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(REPO_ROOT / "data/paper_round1/parsed_clean"),
        "--signoff_csv",
        str(signoff),
        "--protected_csv",
        str(protected),
        "--audit_dir",
        str(audit_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 2
    assert "production parsed_clean" in result.stderr


def test_non_dry_run_writes_experimental_parsed_json_and_audit(tmp_path: Path) -> None:
    result = run_cli(tmp_path)

    assert result.returncode == 0, result.stderr
    output_doc = tmp_path / "output/doc_cli.json"
    assert output_doc.exists()
    data = json.loads(output_doc.read_text(encoding="utf-8"))
    cleaned = data["pages"][0]["blocks"][1]
    assert cleaned["type"] == "paragraph"
    assert cleaned["metadata"]["original_block_type"] == "table_caption"
    assert (tmp_path / "audit/cleanup_audit.csv").exists()
    assert (tmp_path / "audit/protected_caption_check.csv").exists()
