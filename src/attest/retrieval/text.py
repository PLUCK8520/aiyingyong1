"""T3.1 · 本地文档装载与语义分块（md / txt / pdf）。

分块约束来自 T3.3：备用 rerank（qwen3-rerank）**单条上限 4000 token 且超限直接 HTTP 400**，
所以 chunk 必须远小于 4000 token。默认 800 字符 → CJK 约 800 token，留足余量。

分块策略是"段落优先"：先按空行切段，再贪心装桶，避免把一句话从中间切断；
单段超长（无空行的长文）才做硬切 + overlap。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_path: str  # 相对 docs 根目录，便于引用里显示可读路径
    title: str
    ordinal: int
    text: str


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # 纯 Python，对 3.13 最稳（P-1 已实测可导入）

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


_TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def doc_title(path: Path, text: str) -> str:
    m = _TITLE_RE.search(text or "")
    return m.group(1).strip() if m else path.stem


def chunk_text(text: str, *, size: int = 800, overlap: int = 120) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    chunks: list[str] = []
    buf = ""
    step = max(1, size - overlap)
    for para in paras:
        if len(para) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(para), step):
                piece = para[i : i + size].strip()
                if piece:
                    chunks.append(piece)
            continue
        if not buf:
            buf = para
        elif len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}"
        else:
            chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def iter_documents(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_chunks(root: Path, *, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """把 `root` 下所有支持格式的文档切成 chunk。chunk_id 由「相对路径#序号」哈希而来，稳定可复现。"""
    out: list[Chunk] = []
    for path in iter_documents(root):
        text = read_text(path)
        title = doc_title(path, text)
        rel = path.relative_to(root).as_posix()
        for i, piece in enumerate(chunk_text(text, size=size, overlap=overlap)):
            cid = hashlib.sha1(f"{rel}#{i}".encode("utf-8")).hexdigest()[:12]
            out.append(
                Chunk(chunk_id=cid, source_path=rel, title=title, ordinal=i, text=piece)
            )
    return out
