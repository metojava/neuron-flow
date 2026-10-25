#!/usr/bin/env python3
"""Convert the book's HTML chapters to Markdown files in md/."""

import re
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "md"
OUT.mkdir(exist_ok=True)


def inline_md(el) -> str:
    """Convert an element's inline children to markdown."""
    parts: list[str] = []
    for child in el.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            name = child.name
            if name in ("strong", "b"):
                inner = inline_md(child).strip()
                parts.append(f"**{inner}**")
            elif name in ("em", "i"):
                inner = inline_md(child).strip()
                parts.append(f"*{inner}*")
            elif name == "code":
                text = child.get_text()
                parts.append(f"`{text}`")
            elif name == "a":
                href = child.get("href", "")
                text = inline_md(child).strip()
                parts.append(f"[{text}]({href})")
            elif name == "br":
                parts.append("\n")
            elif name == "span":
                parts.append(inline_md(child))
            else:
                parts.append(inline_md(child))
    text = "".join(parts)
    # collapse runs of spaces (but not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text


def table_to_md(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [inline_md(td).strip().replace("\n", " ") for td in tr.find_all(["th", "td"])]
        rows.append(cells)
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    # pad short rows
    for r in rows:
        while len(r) < ncols:
            r.append("")
    header = rows[0]
    body = rows[1:]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def cards_to_md(div: Tag) -> str:
    lines = []
    for a in div.find_all("a", recursive=False):
        strong = a.find("strong")
        span = a.find("span")
        href = a.get("href", "")
        title = strong.get_text(strip=True) if strong else ""
        desc = span.get_text(strip=True) if span else ""
        lines.append(f"- **[{title}]({href})** — {desc}")
    return "\n".join(lines)


def flow_to_md(div: Tag) -> str:
    steps = []
    for child in div.children:
        if isinstance(child, Tag):
            if child.name == "span":
                # main word + optional <small> subtitle, joined with a space
                small = child.find("small")
                small_text = small.get_text(strip=True) if small else ""
                # get full text then remove small's text to get the main word
                full = child.get_text(" ", strip=True)
                main = full
                if small_text and full.endswith(small_text):
                    main = full[: -len(small_text)].strip()
                step = f"**{main}**" if not small_text else f"**{main}** ({small_text})"
                steps.append(step)
            elif child.name == "b":
                steps.append("→")
    return " ".join(steps)


def note_to_md(div: Tag, kind: str = "note") -> str:
    # First strong or plain text
    strong = div.find("strong")
    body = inline_md(div).strip()
    return f"> {body}"


def pre_to_md(pre: Tag) -> str:
    code = pre.find("code")
    text = code.get_text() if code else pre.get_text()
    # strip one leading newline if present
    text = text.lstrip("\n")
    return f"```\n{text.rstrip()}\n```"


def convert_article(article: Tag) -> str:
    out: list[str] = []
    for child in article.children:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name

        if name == "h1":
            out.append(f"# {inline_md(child).strip()}\n")
        elif name == "h2":
            out.append(f"## {inline_md(child).strip()}\n")
        elif name == "h3":
            out.append(f"### {inline_md(child).strip()}\n")
        elif name == "h4":
            out.append(f"#### {inline_md(child).strip()}\n")
        elif name == "p":
            classes = child.get("class", [])
            text = inline_md(child).strip()
            if "lead" in classes:
                out.append(f"*{text}*\n")
            else:
                out.append(f"{text}\n")
        elif name == "pre":
            out.append(pre_to_md(child) + "\n")
        elif name == "table":
            out.append(table_to_md(child) + "\n")
        elif name == "ul":
            items = []
            for li in child.find_all("li", recursive=False):
                items.append(f"- {inline_md(li).strip()}")
            out.append("\n".join(items) + "\n")
        elif name == "ol":
            items = []
            for i, li in enumerate(child.find_all("li", recursive=False), 1):
                items.append(f"{i}. {inline_md(li).strip()}")
            out.append("\n".join(items) + "\n")
        elif name == "div":
            classes = child.get("class", [])
            if "eyebrow" in classes:
                # skip — navigation chrome, duplicated by the H1
                continue
            elif "cards" in classes:
                out.append(cards_to_md(child) + "\n")
            elif "flow" in classes:
                out.append(flow_to_md(child) + "\n")
            elif "note" in classes or "warning" in classes or "tip" in classes:
                out.append(note_to_md(child) + "\n")
            elif "paging" in classes:
                links = child.find_all("a")
                parts = [f"[{inline_md(a).strip()}]({a.get('href','')})" for a in links]
                if parts:
                    out.append("---\n\n" + " · ".join(parts) + "\n")
            else:
                # generic div — recurse-ish: just get its text inline
                text = inline_md(child).strip()
                if text:
                    out.append(text + "\n")
        elif name == "blockquote":
            text = inline_md(child).strip()
            out.append(f"> {text}\n")
        elif name == "hr":
            out.append("---\n")
        elif name == "section":
            text = inline_md(child).strip()
            if text:
                out.append(text + "\n")
        else:
            text = inline_md(child).strip()
            if text:
                out.append(text + "\n")

    # join with blank lines between blocks, collapse excess blanks
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def title_from_html(soup: BeautifulSoup, fallback: str) -> str:
    t = soup.find("title")
    if t and t.get_text(strip=True):
        return t.get_text(strip=True)
    # derive a nicer title from the filename: "16-autograd" -> "16 · Autograd"
    stem = fallback
    m = re.match(r"^(\d+)-(.+)$", stem)
    if m:
        num, rest = m.groups()
        return f"{num} · {rest.replace('-', ' ').title()}"
    return stem.replace("-", " ").title()


def process_file(src: Path, dest: Path):
    html = src.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if article is None:
        # stub page (e.g. 16-autograd.html is just <h1>name</h1>)
        title = title_from_html(soup, src.stem)
        dest.write_text(f"# {title}\n\n*Placeholder chapter — content coming soon.*\n", encoding="utf-8")
        print(f"  OK (stub): {src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")
        return

    # Build a markdown header from title + top/eyebrow
    full_title = title_from_html(soup, src.stem)
    md_body = convert_article(article)

    # Use the article's own H1 if present; otherwise fall back to the page title.
    if md_body.lstrip().startswith("# "):
        final = md_body
    else:
        final = f"# {full_title}\n\n{md_body}"

    # Rewrite internal .html links to .md (leave external/absolute URLs alone)
    final = re.sub(
        r'\((?!https?://|#|mailto:)([^)]+?)\.html(#[^)]*)?\)',
        r'(\1.md\2)',
        final,
    )

    dest.write_text(final, encoding="utf-8")
    print(f"  OK: {src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")


def main():
    files = sorted(ROOT.glob("*.html")) + sorted((ROOT / "projects").glob("*.html"))
    for src in files:
        if src.parent.name == "projects":
            dest = OUT / "projects" / src.name.replace(".html", ".md")
            dest.parent.mkdir(exist_ok=True)
        else:
            dest = OUT / src.name.replace(".html", ".md")
        process_file(src, dest)
    print(f"\nDone. {len(files)} files processed.")


if __name__ == "__main__":
    main()
