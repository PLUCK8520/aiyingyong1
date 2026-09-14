"""T8.1 · 本地知识库管理（Web 端）：列出 / 上传 / 删除 / 重建（2026-09-13）。

**为什么要有这一层**：P3 起 `scout_local` 就一直在跑，但**入库只能敲 CLI**
（`scripts/ingest_local.py`），Web 端没有任何入口。结果是：用户问一个本地语料覆盖不到
的问题，系统只能拒编——而用户并不知道"只要把自己的资料传进去就有真报告了"。
本模块把"入库"从一次性脚本变成日常可用的能力。

**分层归属**：③ 能力层（`attest.*`）。`app/main.py` 只做协议适配（收字节、返 JSON），
"什么算可入库的文档、入库后索引如何保持一致"这类业务判断全在这里——
接入层不碰模型、不碰文件系统语义（架构设计 §2）。

**与 `scripts/ingest_local.py` 的分工**（复用底层，不重复实现）：
    - 分块 → `retrieval.text.load_chunks`。**必须是同一份实现**：列表里显示的 chunk 数
      就是检索层实际索引的条数。若这里另写一套统计，两处迟早漂移，用户会看到
      "我传了 3 个文档，列表说 12 块，检索却召回 0 条"这种无从排查的现象。
    - 向量写入 / 删除 → `retrieval.stores.VectorStore` 协议（numpy 与 chroma 都实现了 delete）。
    - embedding → `llm.gateway.LLMGateway`。
    - `ingest_local.py` 保留为**验收工具**：全量重建 + 回环校验 + onnx 静默下载检测，
      面向"索引是否真的可用"的一次性取证；本模块面向"用户日常维护语料"。职责不重叠。

⚠️ **两种向量后端的入库语义不同，这是本模块最容易写错的地方**：
    - `numpy`（当前默认）：索引**不持久化**。每次检索都由 `build_local_index` 从文档目录
      全量重建 → 上传/删除后**只需 `clear_local_cache()`**，不需要手写任何向量。
    - `chroma`：索引持久化，且 `build_local_index` **仅在 collection 为空时**才写入 →
      上传后**必须手动增量 upsert**，删除后**必须手动 delete**。
      漏了这一步的后果是"文件已删、检索还能召回"——索引与磁盘不一致，且不报错。

⚠️ **删文件用 `unlink()` 而不是移进回收站**：默认目录是 `data/fixtures/local/`，
属**项目数据区**而非用户个人文档区；语义上这是"从知识库里摘除一条语料"，不是"清理磁盘"。
误删的代价限于本地语料，不会波及用户目录之外的文件。

⚠️ **入库的文档会进入 git 工作区**：`data/index/`、`data/cache/`、`data/trace/` 已 gitignore，
但 `data/fixtures/local/` 是**刻意入库**的（那 7 个 `local_*.md` 示例语料是仓库资产，
用户 clone 下来就能体验本地检索）。所以用户上传的资料会出现在 `git status` 里——
这是有意为之（可随仓库带到另一台机器），但**上传敏感资料前必须知道这一点**。
未来若加"上传目录可配置"，建议把用户语料目录与示例语料目录分开。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .config import DATA_DIR, Settings
from .logging import get_logger
from .retrieval.local_search import clear_local_cache
from .retrieval.text import (
    SUPPORTED_SUFFIXES,
    Chunk,
    iter_documents,
    load_chunks,
    read_text,
)

log = get_logger(__name__)

#: 单文件上限。本地知识库是"你自己的资料"，不是数据仓库——
#: 8MB 足够放下几十万字的长报告/论文，超过这个量级更该先做清洗与切分再入库。
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

#: 文件名清洗：Windows 保留字符 + 控制字符。`/` 与 `\` 也在此列（路径分隔）。
_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

#: Windows 保留设备名（写成这些名字的文件在 Windows 上无法创建/删除）
_WIN_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_filename(raw: str) -> str:
    """清洗上传文件名——**这是安全边界，不是体验优化**。

    `Path(docs_dir) / "../../.env"` 会写到项目根目录之外；`C:\\Windows\\x.md`
    在 Windows 上会被当作绝对路径直接落到系统目录。上传入口接受的是**外部输入**，
    必须在落盘前收敛到"当前目录下的一个普通文件名"。

    规则：
      ① 反斜杠统一为斜杠后**只取最后一段**（`../a.md` → `a.md`）；
      ② 剔除控制字符与 Windows 保留字符（`:` `*` `?` `"` `<` `>` `|`）→ 替换为 `_`；
      ③ 去掉首尾空白与点（`..` / `.hidden` 这类形态归零）；
      ④ Windows 保留设备名加前缀（`CON.md` 在 Windows 上创建会失败）；
      ⑤ 文件名过长时截断主体，**保留扩展名**。
    """
    name = (raw or "").replace("\\", "/").split("/")[-1]
    name = _UNSAFE_CHARS.sub("_", name).strip().strip(".")
    if not name:
        raise ValueError("文件名无效（清洗后为空）")

    stem, dot, suffix = name.rpartition(".")
    if not dot:  # 无扩展名：整体当 stem
        stem, suffix = name, ""
    stem = stem.strip().strip(".")
    if stem.upper() in _WIN_RESERVED:
        stem = f"_{stem}"
    if len(stem) > 80:
        stem = stem[:80]
    if not stem:
        raise ValueError("文件名无效（主体为空）")
    return f"{stem}.{suffix}" if suffix else stem


def _docs_dir(settings: Settings) -> Path:
    return Path(settings.local_docs_dir)


def _embedder_name(settings: Settings) -> str:
    """与 `graph/build.py::make_local_search_client` **必须保持一致**。

    两处不一致的后果：上传时按 A 名字写进 collection 元数据，检索时按 B 名字校验
    → `ChromaVectorStore` 抛"向量空间与配置不符"，本地检索整个失效。
    """
    return "mock-hashing" if settings.llm_mode == "mock" else settings.model_embed


def _load_all_chunks(settings: Settings) -> list[Chunk]:
    """全量分块一次，供统计使用。

    单文件上传/删除时也走这里（而不是只算那一个文件）：文档目录通常是几十到几百个
    纯文本文件，全量分块是毫秒级 CPU 操作；换来的是**列表里的数字必然等于索引里的条数**
    ——因为用的是同一个函数、同一份参数。宁可多花几毫秒，也不要两个数字对不上。
    """
    return load_chunks(
        _docs_dir(settings),
        size=settings.local_chunk_chars,
        overlap=settings.local_chunk_overlap,
    )


def _gateway(settings: Settings) -> Any:
    """建一个**只用于入库**的 LLM 网关（embedding 用）。

    不共用 `graph` 里的实例：那个是每个 run 建一次、与 run 的 trace 绑定的
    （见 `SessionManager._build_app`）。入库是独立动作，自己带一个 trace 更清晰。
    """
    from .llm.gateway import LLMGateway
    from .trace.events import TraceWriter

    settings.trace_dir.mkdir(parents=True, exist_ok=True)
    trace = TraceWriter(path=settings.trace_dir / "trace.jsonl", run_id="kb")
    return LLMGateway(settings=settings, trace=trace)


def _chroma_store(settings: Settings, *, dim: int) -> Any:
    """打开（必要时创建）docs collection。仅在 `local_store == "chroma"` 时调用。"""
    from .retrieval.stores import ChromaVectorStore

    return ChromaVectorStore(
        path=Path(settings.local_chroma_dir),
        collection="docs",
        embedder=_embedder_name(settings),
        dim=dim,
    )


def _embed(settings: Settings, texts: list[str]) -> tuple[list[list[float]], int]:
    """调 embedding，返回 (向量, 维度)。"""
    gw = _gateway(settings)
    try:
        vecs, _, _ = gw.embed(list(texts), task="ingest")
    finally:
        try:
            gw.trace.close()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - trace 关闭失败不该影响入库结果
            pass
    dim = len(vecs[0]) if vecs else 0
    return vecs, dim


def _chunk_metadata(settings: Settings, chunks: list[Chunk], *, dim: int) -> list[dict[str, Any]]:
    """与 `build_local_index` 写入的元数据结构保持一致（检索层靠这些键回查来源）。"""
    embedder = _embedder_name(settings)
    return [
        {
            "chunk_id": c.chunk_id,
            "source_path": c.source_path,
            "title": c.title,
            "ordinal": c.ordinal,
            "embedder": embedder,
            "dim": dim,
        }
        for c in chunks
    ]


def _sync_chroma_upsert(settings: Settings, chunks: list[Chunk]) -> int:
    """chroma 档：把新文档的 chunk 增量写进持久索引。返回写入条数。"""
    if not chunks:
        return 0
    vecs, dim = _embed(settings, [c.text for c in chunks])
    if not vecs:
        log.warning("[kb] embedding 返回空，跳过持久索引写入（下次检索会走内存重建）")
        return 0
    store = _chroma_store(settings, dim=dim)
    store.upsert(
        [c.chunk_id for c in chunks],
        vecs,
        [c.text for c in chunks],
        _chunk_metadata(settings, chunks, dim=dim),
    )
    return len(chunks)


def _stats_from(
    settings: Settings, files: list[Path], chunks: list[Chunk], docs_dir: Path
) -> dict[str, Any]:
    """装配统计信息（`GET /api/kb` 与上传/删除响应共用）。"""
    return {
        "documents": len(files),
        "chunks": len(chunks),
        "docs_dir": str(docs_dir),
        "store": settings.local_store,
        "embedder": _embedder_name(settings),
        "embed_fallback": settings.embed_fallback,
        "chunk_chars": settings.local_chunk_chars,
        "chunk_overlap": settings.local_chunk_overlap,
        # 下面两项决定"这个知识库能不能支撑出一份真报告"，必须显式暴露给前端：
        # 本地检索关掉 = 传了也白传；web 检索是 mock = 只有本地这一条证据来源。
        "local_enabled": settings.local_enabled,
        "search_mode": settings.search_mode,
        "web_search_ready": settings.search_mode == "tavily" and bool(settings.tavily_api_key),
        "llm_mode": settings.llm_mode,
    }


def describe(settings: Settings) -> dict[str, Any]:
    """知识库全貌：文档清单 + 统计（`GET /api/kb` 的数据源）。"""
    docs_dir = _docs_dir(settings)
    files = iter_documents(docs_dir)
    chunks = _load_all_chunks(settings)

    n_by_file: dict[str, int] = {}
    title_by_file: dict[str, str] = {}
    for c in chunks:
        n_by_file[c.source_path] = n_by_file.get(c.source_path, 0) + 1
        title_by_file.setdefault(c.source_path, c.title)

    items: list[dict[str, Any]] = []
    for p in files:
        rel = p.relative_to(docs_dir).as_posix()
        st = p.stat()
        items.append(
            {
                "name": rel,
                "title": title_by_file.get(rel) or p.stem,
                "size": st.st_size,
                "chunks": n_by_file.get(rel, 0),
                "modified": st.st_mtime,
                # 仓库自带的 7 个示例语料（`local_*.md`）。标出来，否则用户会纳闷
                # "我一个都没传，怎么已经有 7 个文档了"。
                "sample": rel.startswith("local_"),
            }
        )
    items.sort(key=lambda x: (not x["sample"], x["name"]))
    return {"documents": items, "stats": _stats_from(settings, files, chunks, docs_dir)}


def ingest_document(settings: Settings, filename: str, data: bytes) -> dict[str, Any]:
    """把一个上传的文档写入知识库。

    流程：清洗文件名 → 校验类型与大小 → 落盘 → **立刻验证可解析** → 同步持久索引
    → 失效检索缓存。

    第 4 步是关键：损坏的 PDF、扫描件（无文本层）、空文件如果直接入库，会变成
    "在列表里占位、永远召回不到内容"的僵尸文档——用户以为传成功了，实际什么都没发生。
    验证失败**回滚落盘**并如实报错。
    """
    if not data:
        raise ValueError("文件内容为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"文件过大：{len(data) / 1048576:.1f}MB，上限 {MAX_UPLOAD_BYTES // 1048576}MB"
        )

    safe = safe_filename(filename)
    suffix = Path(safe).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = " / ".join(sorted(s.lstrip(".") for s in SUPPORTED_SUFFIXES))
        raise ValueError(f"不支持的格式：{suffix or '(无扩展名)'}，仅支持 {supported}")

    docs_dir = _docs_dir(settings)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / safe
    replaced = dest.exists()
    dest.write_bytes(data)

    try:
        text = read_text(dest)
    except Exception as exc:  # noqa: BLE001 - 解析失败要给中文提示，不能让堆栈冒到前端
        dest.unlink(missing_ok=True)
        raise ValueError(f"文档无法解析（{type(exc).__name__}）：{exc}") from exc
    if not text.strip():
        dest.unlink(missing_ok=True)
        raise ValueError("文档里提取不到文本（扫描件 PDF 需要先做 OCR 才能入库）")

    chunks = [c for c in _load_all_chunks(settings) if c.source_path == safe]

    # 先失效缓存再写持久索引：`clear_local_cache()` 只清 Python 侧的 LocalIndex，
    # 若 chroma 的 upsert 失败，下次检索仍会从磁盘文档重建（数据不丢），
    # 只是持久索引短暂落后——比"缓存里留着旧索引"更安全。
    clear_local_cache()
    synced = 0
    if settings.local_store == "chroma":
        synced = _sync_chroma_upsert(settings, chunks)

    log.info(
        f"[kb] 入库 {safe}（{len(data)}B → {len(chunks)} chunk）"
        f"{'，覆盖同名文档' if replaced else ''}，持久索引写入 {synced} 条"
    )
    return {
        "name": safe,
        "size": len(data),
        "chunks": len(chunks),
        "replaced": replaced,
        "store_synced": synced,
        "stats": _stats_from(settings, iter_documents(docs_dir), _load_all_chunks(settings), docs_dir),
    }


def remove_document(settings: Settings, name: str) -> dict[str, Any]:
    """从知识库摘除一个文档（删文件 + 同步持久索引 + 失效缓存）。"""
    docs_dir = _docs_dir(settings)
    safe = safe_filename(name)
    target = docs_dir / safe
    # 纵深防御：safe_filename 已挡住目录穿越，这里再确认一次"落点确实在库内"，
    # 因为下一行就要删文件——这类操作值得多一道校验。
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"知识库中没有这个文档：{safe}")

    rel = target.relative_to(docs_dir).as_posix()
    # 先算出这个文件对应哪些 chunk（必须在删文件之前算：chunk_id 由「相对路径#序号」哈希而来，
    # 文件没了就算不出来了）。
    ids = [c.chunk_id for c in _load_all_chunks(settings) if c.source_path == rel]

    target.unlink()

    unlinked = 0
    if settings.local_store == "chroma" and ids:
        try:
            store = _chroma_store(settings, dim=0)  # dim=0 = 不校验维度，只做删除
            store.delete(ids)
            unlinked = len(ids)
        except Exception as exc:  # noqa: BLE001
            # 删索引失败不能让整个请求失败：文件已经删了，如实告知"索引里可能残留"，
            # 用户可用重建（POST /api/kb/rebuild）兜底。**不谎报成功**。
            log.warning(f"[kb] 删除 {rel} 的持久索引失败：{type(exc).__name__}: {exc}")
            raise ValueError(
                f"文件已删除，但从向量索引摘除失败（{type(exc).__name__}）。"
                f"该文档可能仍会被检索命中，请执行「重建索引」修复。"
            ) from exc

    clear_local_cache()
    log.info(f"[kb] 移除 {rel}（同步摘除索引 {unlinked} 条）")
    return {
        "name": rel,
        "removed_chunks": len(ids),
        "store_unlinked": unlinked,
        "stats": _stats_from(settings, iter_documents(docs_dir), _load_all_chunks(settings), docs_dir),
    }


def rebuild_index(settings: Settings) -> dict[str, Any]:
    """丢弃持久索引并失效缓存，逼下次检索从文档目录全量重建。

    什么时候需要：换了 embedding 模型（`ChromaVectorStore` 会因 embedder/dim
    不匹配直接抛错，这是 NFR-11 的设计），或索引与磁盘出现不一致。

    ⚠️ 这里会 `shutil.rmtree` ——所以先断言目标目录在 `data/` 之内。
    路径来自配置而非外部输入，但**破坏性操作值得多一道约束**。
    """
    clear_local_cache()
    removed = False
    index_dir = Path(settings.local_chroma_dir).resolve()
    data_root = DATA_DIR.resolve()

    if settings.local_store != "chroma":
        log.info("[kb] 当前是内存向量档（numpy），索引本就不持久化；已仅失效缓存")
        return {"rebuilt": True, "removed_index": False, "store": settings.local_store}

    if index_dir == data_root or data_root not in index_dir.parents:
        raise ValueError(f"拒绝删除：索引目录不在 data/ 之内（{index_dir}）")
    if index_dir.exists():
        shutil.rmtree(index_dir)
        removed = True

    log.info(f"[kb] 已丢弃持久索引（{index_dir}）；下次检索将全量重建")
    return {"rebuilt": True, "removed_index": removed, "store": settings.local_store}


__all__ = [
    "MAX_UPLOAD_BYTES",
    "describe",
    "ingest_document",
    "rebuild_index",
    "remove_document",
    "safe_filename",
]
