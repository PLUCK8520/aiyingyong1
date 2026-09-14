"""T8.1 / T8.2 · 知识库管理的回归测试。

**这一组要证明的**（不是"跑过就算"）：
  1. `safe_filename` 真的挡得住目录穿越——上传入口接受的是**外部输入**，
     `Path(docs_dir) / "../../.env"` 会写到项目外；这是安全边界，必须有测试钉住；
  2. 上传的文档**立刻**能被本地检索召回（含"入库后必须失效进程级索引缓存"这个易漏点）；
  3. 入库前的可解析性校验失败会**回滚落盘**——否则留下"列表里占位、永远召回不到"的僵尸文档；
  4. 删除同时摘掉索引（numpy 档下索引由磁盘重建，故只需失效缓存）；
  5. 接口层契约：上传走 `X-Filename` 头（不用 multipart）、错误码语义正确。

背景：P3 起 `scout_local` 一直在跑，但入库只能敲 CLI，Web 端没有任何入口。
用户问一个语料覆盖不到的问题会被拒编——拒编本身是对的，但用户不知道
"把自己的资料传进去就能出真报告"。这一组测试保护的就是那个入口。
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from attest.config import Settings
from attest.kb import (
    MAX_UPLOAD_BYTES,
    describe,
    ingest_document,
    rebuild_index,
    remove_document,
    safe_filename,
)
from attest.llm.gateway import LLMGateway
from attest.retrieval.local_search import clear_local_cache, make_local_client
from attest.retrieval.text import iter_documents
from attest.trace.events import TraceWriter


def _settings(tmp_path: Path) -> Settings:
    s = Settings(
        llm_mode="mock",
        search_mode="mock",
        local_enabled=True,
        local_docs_dir=tmp_path / "kb",
        local_chroma_dir=tmp_path / "chroma",
        local_store="numpy",
        trace_dir=tmp_path / "trace",
        report_dir=tmp_path / "reports",
        fixture_dir=tmp_path / "fixtures",
        checkpoint_db=tmp_path / "cp.sqlite",
        profile_db=tmp_path / "profile.sqlite",
    )
    s.ensure_dirs()
    s.local_docs_dir.mkdir(parents=True, exist_ok=True)
    clear_local_cache()  # 进程级缓存的 key 含目录名，tmp 路径唯一；仍清一次保证隔离
    return s


def _embed_fn(settings: Settings):
    trace = TraceWriter(path=settings.trace_dir / "trace.jsonl", run_id="kb-test")
    gw = LLMGateway(settings=settings, trace=trace)
    return lambda texts: gw.embed(list(texts), task="kb-test")[0]


# ============================================================ 1. 文件名安全


@pytest.mark.parametrize(
    "raw",
    [
        "../../.env",
        "..\\..\\x.txt",
        "C:\\Windows\\evil.md",
        "/etc/passwd.txt",
        "....//....//a.md",
        "a/b/c/d.md",
    ],
)
def test_safe_filename_blocks_path_traversal(raw: str) -> None:
    """任何形态的路径输入都必须收敛成"一个普通文件名"。"""
    out = safe_filename(raw)
    assert "/" not in out and "\\" not in out, f"{raw!r} → {out!r} 仍含路径分隔符"
    assert ".." not in out, f"{raw!r} → {out!r} 仍含上跳"
    assert out and not out.startswith("."), f"{raw!r} → {out!r}"


def test_safe_filename_escapes_windows_reserved_names() -> None:
    """`CON.md` 这类保留名在 Windows 上创建会失败——必须改名而不是让用户撞墙。"""
    assert safe_filename("CON.md") == "_CON.md"
    assert safe_filename("nul.txt") == "_nul.txt"
    assert safe_filename("com1.md").startswith("_")


def test_safe_filename_rejects_empty_after_cleaning() -> None:
    for raw in ("", "...", "   ", "///"):
        with pytest.raises(ValueError):
            safe_filename(raw)


def test_safe_filename_keeps_chinese_and_extension() -> None:
    assert safe_filename("我的调研笔记.md") == "我的调研笔记.md"


def test_safe_filename_truncates_long_stem_but_keeps_suffix() -> None:
    out = safe_filename("x" * 300 + ".md")
    assert out.endswith(".md")
    assert len(out) <= 84


# ============================================================ 2. 入库校验


def test_ingest_rejects_unsupported_suffix(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="不支持"):
        ingest_document(settings, "payload.exe", b"MZ\x90\x00")


def test_ingest_rejects_empty_and_oversize(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="为空"):
        ingest_document(settings, "a.md", b"")
    with pytest.raises(ValueError, match="过大"):
        ingest_document(settings, "a.md", b"x" * (MAX_UPLOAD_BYTES + 1))


def test_ingest_rolls_back_when_unparsable(tmp_path: Path) -> None:
    """解析失败必须回滚：留下僵尸文档比直接报错更糟——用户以为传成功了。"""
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="无法解析"):
        ingest_document(settings, "坏文件.pdf", b"this is definitely not a pdf")
    assert list(iter_documents(settings.local_docs_dir)) == [], "失败的文件不得留在磁盘上"


def test_ingest_rejects_pdf_without_text_layer(tmp_path: Path) -> None:
    """扫描件 PDF（无文本层）要明确拒绝并给出 OCR 提示，而不是入库一个空文档。"""
    settings = _settings(tmp_path)
    # 一个结构合法但零页内容的 PDF
    empty_pdf = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
    )
    with pytest.raises(ValueError):
        ingest_document(settings, "扫描件.pdf", empty_pdf)
    assert list(iter_documents(settings.local_docs_dir)) == []


def test_ingest_overwrites_same_name(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    r1 = ingest_document(settings, "笔记.md", "# 笔记\n\n第一版内容，讲向量检索。".encode())
    assert r1["replaced"] is False
    r2 = ingest_document(settings, "笔记.md", "# 笔记\n\n第二版内容，讲混合检索。".encode())
    assert r2["replaced"] is True
    assert len(list(iter_documents(settings.local_docs_dir))) == 1, "同名应覆盖而不是并存"


# ============================================================ 3. 列表与统计


def test_describe_counts_match_actual_chunks(tmp_path: Path) -> None:
    """列表里的 chunk 数必须等于分块实现给出的条数（同一个函数算出来的）。"""
    settings = _settings(tmp_path)
    ingest_document(settings, "a.md", "# 甲\n\n第一段内容。".encode())
    ingest_document(settings, "b.md", "# 乙\n\n第二段内容。".encode())

    d = describe(settings)
    assert d["stats"]["documents"] == 2
    assert d["stats"]["chunks"] == sum(x["chunks"] for x in d["documents"])
    assert all(x["chunks"] >= 1 for x in d["documents"])
    assert {x["name"] for x in d["documents"]} == {"a.md", "b.md"}


def test_describe_marks_sample_docs(tmp_path: Path) -> None:
    """仓库自带的 `local_*.md` 要标成示例——否则用户会纳闷"我没传过这些"。"""
    settings = _settings(tmp_path)
    ingest_document(settings, "local_market_note.md", "# 示例\n\n内容。".encode())
    ingest_document(settings, "我的资料.md", "# 我的\n\n内容。".encode())
    by_name = {d["name"]: d for d in describe(settings)["documents"]}
    assert by_name["local_market_note.md"]["sample"] is True
    assert by_name["我的资料.md"]["sample"] is False


# ============================================================ 4. 核心：入库即被检索到


def test_uploaded_document_becomes_retrievable(tmp_path: Path) -> None:
    """**这是整个功能的验收核心**：传进去的文档必须立刻能被本地检索召回。

    它同时钉住一个极易漏的点——入库后必须 `clear_local_cache()`：
    `make_local_client` 有**进程级缓存**（key = 目录 + embedder + 后端），
    不失效缓存的话，检索会一直用旧索引，表现为"文件在、列表里也有、
    检索就是召回不到"这种最难查的状态。
    """
    settings = _settings(tmp_path)
    embed = _embed_fn(settings)
    dim = len(embed(["dim probe"])[0])

    def client():
        return make_local_client(
            docs_dir=settings.local_docs_dir,
            embed_fn=embed,
            embedder="mock-hashing",
            dim=dim,
            use_chroma=False,
            chunk_chars=settings.local_chunk_chars,
            chunk_overlap=settings.local_chunk_overlap,
        )

    assert client().search("量子比特 退相干", max_results=3) == [], "空库不该召回任何东西"

    ingest_document(
        settings,
        "量子计算笔记.md",
        "# 量子计算工程瓶颈\n\n当前的主要瓶颈是量子比特的退相干时间过短，纠错开销极高。".encode(),
    )

    hits = client().search("量子比特 退相干", max_results=3)
    assert hits, "上传后必须立刻可召回（检查 clear_local_cache 是否生效）"
    assert any("退相干" in h.content for h in hits)

    remove_document(settings, "量子计算笔记.md")
    assert client().search("量子比特 退相干", max_results=3) == [], "删除后不该再召回"


# ============================================================ 5. 删除


def test_remove_missing_document_raises_file_not_found(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(FileNotFoundError):
        remove_document(settings, "不存在.md")


def test_remove_blocks_traversal(tmp_path: Path) -> None:
    """删除接口同样吃外部输入——穿越尝试应当被收敛成"库内不存在"而不是删到别处。"""
    settings = _settings(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("# 库外文件\n\n不该被删。", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        remove_document(settings, "../outside.md")
    assert outside.exists(), "库外文件必须完好无损"


def test_remove_reports_chunk_count(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ingest_document(settings, "a.md", "# 甲\n\n一段内容。".encode())
    r = remove_document(settings, "a.md")
    assert r["name"] == "a.md"
    assert r["removed_chunks"] >= 1
    assert r["stats"]["documents"] == 0


# ============================================================ 6. 重建


def test_rebuild_does_not_touch_documents_on_numpy_store(tmp_path: Path) -> None:
    """numpy 档下索引本就不持久化，重建只能清缓存——**绝不能**误删文档目录。"""
    settings = _settings(tmp_path)
    ingest_document(settings, "a.md", "# 甲\n\n内容。".encode())
    r = rebuild_index(settings)
    assert r["removed_index"] is False
    assert list(iter_documents(settings.local_docs_dir)), "文档目录被误删了"


# ============================================================ 7. 接口层契约


def _api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把服务端配置全部指向 tmp，再起一个 TestClient（会触发 lifespan 装配）。"""
    monkeypatch.setenv("ATTEST_LOCAL_DOCS", str(tmp_path / "kb"))
    monkeypatch.setenv("ATTEST_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("ATTEST_TRACE_DIR", str(tmp_path / "trace"))
    monkeypatch.setenv("ATTEST_REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("ATTEST_CHECKPOINT_DB", str(tmp_path / "cp.sqlite"))
    monkeypatch.setenv("ATTEST_PROFILE_DB", str(tmp_path / "profile.sqlite"))
    monkeypatch.setenv("ATTEST_FIXTURE_DIR", str(tmp_path / "fixtures"))
    monkeypatch.setenv("ATTEST_LLM_MODE", "mock")
    monkeypatch.setenv("ATTEST_SEARCH_MODE", "mock")

    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app())


def test_api_kb_upload_list_delete_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _api_client(tmp_path, monkeypatch) as client:
        r = client.get("/api/kb")
        assert r.status_code == 200
        assert r.json()["documents"] == []

        # 文件名走 URL 编码的自定义头（不用 multipart：见 app/main.py 的说明）
        body = "# 我的调研笔记\n\n混合检索把 BM25 与向量召回用 RRF 融合。".encode()
        r = client.post(
            "/api/kb/upload",
            content=body,
            headers={"X-Filename": quote("我的调研笔记.md")},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["name"] == "我的调研笔记.md"
        assert payload["chunks"] >= 1

        r = client.get("/api/kb")
        names = [d["name"] for d in r.json()["documents"]]
        assert "我的调研笔记.md" in names

        r = client.delete(f"/api/kb/{quote('我的调研笔记.md')}")
        assert r.status_code == 200, r.text
        assert r.json()["removed_chunks"] >= 1

        assert client.get("/api/kb").json()["documents"] == []


def test_api_kb_upload_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _api_client(tmp_path, monkeypatch) as client:
        # 缺 X-Filename → 422
        r = client.post("/api/kb/upload", content=b"# x")
        assert r.status_code == 422

        # 不支持的格式 → 422（且文案可读）
        r = client.post("/api/kb/upload", content=b"MZ", headers={"X-Filename": "a.exe"})
        assert r.status_code == 422
        assert "不支持" in r.json()["detail"]

        # 删除不存在的文档 → 404
        r = client.delete("/api/kb/nope.md")
        assert r.status_code == 404


def test_api_kb_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _api_client(tmp_path, monkeypatch) as client:
        r = client.post("/api/kb/rebuild")
        assert r.status_code == 200
        assert r.json()["rebuilt"] is True
