"""Deterministic, heading-aware Markdown chunking for parsed papers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}")


@dataclass(frozen=True, slots=True)
class MarkdownChunk:
    """One section-bounded chunk and its heading breadcrumb."""

    ordinal: int
    heading_path: tuple[str, ...]
    content: str


def _section_windows(
    body: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split a section near whitespace while retaining bounded overlap."""
    body = body.strip()
    if not body:
        return []
    windows: list[str] = []
    start = 0
    while start < len(body):
        hard_end = min(len(body), start + max_chars)
        end = hard_end
        if hard_end < len(body):
            lower_bound = start + max(max_chars // 2, 1)
            candidates = [
                body.rfind("\n\n", lower_bound, hard_end),
                body.rfind("\n", lower_bound, hard_end),
                body.rfind(" ", lower_bound, hard_end),
            ]
            end = max(candidates)
            if end < lower_bound:
                end = hard_end
        piece = body[start:end].strip()
        if piece:
            windows.append(piece)
        if end >= len(body):
            break
        next_start = max(end - overlap_chars, start + 1)
        while next_start < end and body[next_start].isspace():
            next_start += 1
        start = next_start
    return windows


def chunk_markdown(
    markdown: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[MarkdownChunk]:
    """Split Markdown without crossing heading-defined section boundaries."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars must be non-negative and smaller than max_chars"
        )

    sections: list[tuple[tuple[str, ...], str]] = []
    heading_stack: list[str] = []
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((tuple(heading_stack), body))
        body_lines.clear()

    for line in markdown.splitlines():
        heading = _HEADING_RE.match(line)
        if heading is None:
            body_lines.append(line)
            continue
        flush()
        level = len(heading.group(1))
        title = heading.group(2).strip()
        heading_stack[:] = heading_stack[: level - 1]
        heading_stack.append(title)
    flush()

    if not sections and markdown.strip():
        sections.append((tuple(), markdown.strip()))

    chunks: list[MarkdownChunk] = []
    for heading_path, section_body in sections:
        full_prefix = " > ".join(heading_path)
        prefix = full_prefix[: max_chars // 3].rstrip()
        body_limit = max(max_chars - len(prefix) - 2, 1)
        for piece in _section_windows(
            section_body,
            max_chars=body_limit,
            overlap_chars=min(overlap_chars, max(body_limit - 1, 0)),
        ):
            content = f"{prefix}\n\n{piece}" if prefix else piece
            chunks.append(
                MarkdownChunk(
                    ordinal=len(chunks),
                    heading_path=heading_path,
                    content=content,
                )
            )
    return chunks


def build_index_probe(markdown: str, fallback: str) -> str:
    """Build a deterministic smoke-test query from document headings."""
    headings = [
        match.group(2).strip()
        for line in markdown.splitlines()
        if (match := _HEADING_RE.match(line)) is not None
    ]
    selected = [heading for heading in headings if len(heading) >= 3][:3]
    if selected:
        return " ".join(selected)[:240]
    words = _WORD_RE.findall(markdown[:4000])[:12]
    return (" ".join(words) or fallback.strip() or "document evidence")[:240]
