from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import httpx


HELP_TEXT = """\
交互命令:
  /help                         显示帮助
  /quit                         退出
  /health                       检查服务健康状态
  /show                         显示当前请求配置
  /reset                        重置 session 和全部过滤条件
  /debug on|off                 切换 include_debug
  /session [session_id]         查看或设置 session_id
  /tenant <tenant_id>           设置 tenant_id
  /docids <a,b,c>               设置 doc_ids
  /sections <a,b,c>             设置 sections
  /sources <a,b,c>              设置 source_files
  /minscore <float|none>        设置 min_score
  /history                      拉取当前 session 的服务端历史

直接输入任意内容会作为 question 发到 /v1/ask。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交互式测试 SynBio RAG API")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="RAG API 地址，默认: http://127.0.0.1:8000",
    )
    parser.add_argument("--session-id", default=None, help="初始会话 ID")
    parser.add_argument("--tenant-id", default="default", help="租户 ID")
    parser.add_argument("--doc-ids", default="", help="初始 doc_ids，逗号分隔")
    parser.add_argument("--sections", default="", help="初始 sections，逗号分隔")
    parser.add_argument("--source-files", default="", help="初始 source_files，逗号分隔")
    parser.add_argument("--min-score", type=float, default=None, help="初始最小分数阈值")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="默认携带 include_debug=true",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP 请求超时秒数，默认 180",
    )
    parser.add_argument(
        "--map",
        default="data/paper_round1_pdf_rename_map.csv",
        help="文档映射表路径，默认: data/paper_round1_pdf_rename_map.csv",
    )
    return parser.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def print_json(title: str, payload: Any) -> None:
    print(f"\n[{title}]")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def load_name_map(mapping_path: str) -> tuple[dict[str, str], dict[str, str]]:
    by_doc_id: dict[str, str] = {}
    by_new_name: dict[str, str] = {}
    path = Path(mapping_path)
    if not path.exists():
        return by_doc_id, by_new_name

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            real_name = (row.get("old_name") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            new_name = (row.get("new_name") or "").strip()
            if real_name and doc_id:
                by_doc_id[doc_id] = real_name
            if real_name and new_name:
                by_new_name[new_name] = real_name
    return by_doc_id, by_new_name


def healthcheck(client: httpx.Client, base_url: str) -> None:
    response = client.get(f"{base_url}/healthz")
    response.raise_for_status()
    print_json("healthz", response.json())


def fetch_history(client: httpx.Client, base_url: str, session_id: str | None) -> None:
    if not session_id:
        print("当前没有 session_id，先发起一轮对话或通过 /session 设置。")
        return
    response = client.get(f"{base_url}/v1/sessions/{session_id}")
    response.raise_for_status()
    data = response.json()
    print_json("session", data)


def build_payload(question: str, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": question,
        "session_id": state["session_id"],
        "tenant_id": state["tenant_id"],
        "doc_ids": state["doc_ids"],
        "sections": state["sections"],
        "source_files": state["source_files"],
        "min_score": state["min_score"],
        "include_debug": state["debug"],
    }


def print_response(
    data: dict[str, Any],
    name_map_by_doc_id: dict[str, str],
    name_map_by_new_name: dict[str, str],
) -> None:
    print(f"\nsession_id: {data.get('session_id')}")
    print(f"route: {data.get('route')} | confidence: {data.get('confidence')}")
    print("\n[answer]")
    print(data.get("answer", ""))

    citations = data.get("citations") or []
    if citations:
        print("\n[citations]")
        for idx, item in enumerate(citations, start=1):
            doc_id = item.get("doc_id") or ""
            source = item.get("source_file") or doc_id or "-"
            original_name = name_map_by_doc_id.get(doc_id) or name_map_by_new_name.get(source)
            section = item.get("section") or "-"
            score = item.get("score")
            if original_name:
                print(
                    f"{idx}. {source} | original={original_name} | "
                    f"section={section} | score={score}"
                )
            else:
                print(f"{idx}. {source} | section={section} | score={score}")

    references = data.get("external_references") or []
    if references:
        print_json("external_references", references)

    debug = data.get("debug") or {}
    if debug:
        print_json("debug", debug)


def handle_command(command: str, state: dict[str, Any], client: httpx.Client, base_url: str) -> bool:
    parts = shlex.split(command)
    op = parts[0].lower()

    if op == "/help":
        print(HELP_TEXT)
        return True
    if op == "/quit":
        raise SystemExit(0)
    if op == "/health":
        healthcheck(client, base_url)
        return True
    if op == "/show":
        print_json("state", state)
        return True
    if op == "/reset":
        state["session_id"] = None
        state["tenant_id"] = "default"
        state["doc_ids"] = []
        state["sections"] = []
        state["source_files"] = []
        state["min_score"] = None
        state["debug"] = False
        print("已重置 session、过滤条件和 debug 开关。")
        print_json("state", state)
        return True
    if op == "/history":
        fetch_history(client, base_url, state["session_id"])
        return True
    if op == "/debug":
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            print("用法: /debug on|off")
            return True
        state["debug"] = parts[1].lower() == "on"
        print(f"include_debug={state['debug']}")
        return True
    if op == "/session":
        if len(parts) == 1:
            print(f"session_id={state['session_id']}")
            return True
        state["session_id"] = parts[1]
        print(f"session_id={state['session_id']}")
        return True
    if op == "/tenant":
        if len(parts) != 2:
            print("用法: /tenant <tenant_id>")
            return True
        state["tenant_id"] = parts[1]
        print(f"tenant_id={state['tenant_id']}")
        return True
    if op == "/docids":
        state["doc_ids"] = parse_csv(parts[1]) if len(parts) > 1 else []
        print(f"doc_ids={state['doc_ids']}")
        return True
    if op == "/sections":
        state["sections"] = parse_csv(parts[1]) if len(parts) > 1 else []
        print(f"sections={state['sections']}")
        return True
    if op == "/sources":
        state["source_files"] = parse_csv(parts[1]) if len(parts) > 1 else []
        print(f"source_files={state['source_files']}")
        return True
    if op == "/minscore":
        if len(parts) != 2:
            print("用法: /minscore <float|none>")
            return True
        state["min_score"] = None if parts[1].lower() == "none" else float(parts[1])
        print(f"min_score={state['min_score']}")
        return True

    print("未知命令，输入 /help 查看可用命令。")
    return True


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    name_map_by_doc_id, name_map_by_new_name = load_name_map(args.map)
    state: dict[str, Any] = {
        "session_id": args.session_id,
        "tenant_id": args.tenant_id,
        "doc_ids": parse_csv(args.doc_ids),
        "sections": parse_csv(args.sections),
        "source_files": parse_csv(args.source_files),
        "min_score": args.min_score,
        "debug": args.debug,
    }

    print(f"base_url={base_url}")
    with httpx.Client(timeout=args.timeout) as client:
        try:
            healthcheck(client, base_url)
        except Exception as exc:
            print(f"healthcheck 失败: {type(exc).__name__}: {exc}")
            print("请先确认 RAG 服务已启动，例如:")
            print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
            return 1

        print(HELP_TEXT)
        print("示例:")
        print("  python scripts/ops/interactive_rag_cli.py")
        print("  python scripts/ops/interactive_rag_cli.py --base-url http://127.0.0.1:8002 --debug")
        print("当前配置:")
        print_json("state", state)

        while True:
            try:
                raw = input("\nrag> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not raw:
                continue
            if raw.startswith("/"):
                try:
                    handle_command(raw, state, client, base_url)
                except ValueError as exc:
                    print(f"参数错误: {exc}")
                continue

            payload = build_payload(raw, state)
            try:
                response = client.post(f"{base_url}/v1/ask", json=payload)
                response.raise_for_status()
            except Exception as exc:
                print(f"请求失败: {type(exc).__name__}: {exc}")
                continue

            data = response.json()
            state["session_id"] = data.get("session_id") or state["session_id"]
            print_response(data, name_map_by_doc_id, name_map_by_new_name)


if __name__ == "__main__":
    sys.exit(main())
