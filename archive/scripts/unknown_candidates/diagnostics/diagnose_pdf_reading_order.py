#!/usr/bin/env python3
import argparse
import fitz  # PyMuPDF


def guess_column(x0, x1, page_width):
    mid = page_width / 2
    center = (x0 + x1) / 2

    # 保守判断：左半区 / 右半区 / 跨栏
    if x0 < mid and x1 <= mid + 30:
        return "L"
    if x0 >= mid - 30 and x1 > mid:
        return "R"
    if x0 < mid and x1 > mid:
        return "SPAN"
    return "UNK"


def clean_text(text, limit=120):
    return " ".join(text.split())[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--pages", default="1", help="例如 1,2,4 或 1-3，页码从 1 开始")
    args = parser.parse_args()

    doc = fitz.open(args.pdf)

    page_numbers = []
    for part in args.pages.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            page_numbers.extend(range(int(a), int(b) + 1))
        else:
            page_numbers.append(int(part))

    for page_no in page_numbers:
        page = doc[page_no - 1]
        page_width = page.rect.width
        blocks = page.get_text("blocks")

        print("=" * 120)
        print(f"PAGE {page_no} / width={page_width:.1f}")
        print("=" * 120)

        rows = []
        for i, b in enumerate(blocks):
            x0, y0, x1, y1, text, block_no, block_type = b[:7]
            text_short = clean_text(text)
            if not text_short:
                continue
            col = guess_column(x0, x1, page_width)
            rows.append({
                "i": i,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "col": col,
                "text": text_short,
            })

        print("\n[原始 PyMuPDF block 顺序]")
        for r in rows:
            print(
                f"{r['i']:03d} col={r['col']:4s} "
                f"x0={r['x0']:6.1f} y0={r['y0']:6.1f} "
                f"x1={r['x1']:6.1f} y1={r['y1']:6.1f} | {r['text']}"
            )

        print("\n[简单 y0,x0 排序顺序]")
        for j, r in enumerate(sorted(rows, key=lambda r: (r["y0"], r["x0"]))):
            print(
                f"{j:03d} orig={r['i']:03d} col={r['col']:4s} "
                f"x0={r['x0']:6.1f} y0={r['y0']:6.1f} | {r['text']}"
            )

        print("\n[双栏 column,y0,x0 排序顺序]")
        col_order = {"L": 0, "SPAN": 1, "R": 2, "UNK": 3}
        for j, r in enumerate(sorted(rows, key=lambda r: (col_order.get(r["col"], 9), r["y0"], r["x0"]))):
            print(
                f"{j:03d} orig={r['i']:03d} col={r['col']:4s} "
                f"x0={r['x0']:6.1f} y0={r['y0']:6.1f} | {r['text']}"
            )


if __name__ == "__main__":
    main()