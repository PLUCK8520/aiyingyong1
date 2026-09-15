# -*- coding: utf-8 -*-
"""把 Google Fonts 的 IBM Plex Sans / JetBrains Mono 拉到本地自托管。

动机（基于 2026-09-15 实测，不是想当然）：
  - 实测 fonts.googleapis.com 与 gstatic woff2 **可达**（CSS 200 / woff2 200）——
    所以**不是"加载不到"**，改造理由要写准：**消除外网依赖、首屏更快、离线可用**。
  - 只取 **latin 子集**：中文不由这两款字体提供（走系统字体栈），
    把 latin/latin-ext/cyrillic/greek/vietnamese 全下下来是白费体积。

产出：frontend/public/fonts/*.woff2 + frontend/src/fonts.css
"""
from __future__ import annotations

import re
from pathlib import Path

import httpx

ROOT = Path(r"D:\aiyingyong1\attest\frontend")
FONT_DIR = ROOT / "public" / "fonts"
CSS_OUT = ROOT / "src" / "fonts.css"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

FAMILIES = {
    "IBM Plex Sans": ("ibm-plex-sans", "https://fonts.googleapis.com/css2"
                      "?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap"),
    "JetBrains Mono": ("jetbrains-mono", "https://fonts.googleapis.com/css2"
                       "?family=JetBrains+Mono:wght@400;500&display=swap"),
}

# Google Fonts 的 CSS 形如：
#   /* latin */
#   @font-face { ... src: url(https://...woff2) format('woff2'); unicode-range: ...; }
_FACE_RE = re.compile(
    r"/\*\s*(?P<subset>[\w-]+)\s*\*/\s*@font-face\s*\{(?P<body>[^}]+)\}", re.DOTALL
)
_URL_RE = re.compile(r"url\((https://[^)]+\.woff2)\)")
_WEIGHT_RE = re.compile(r"font-weight:\s*(\d+)")
_STYLE_RE = re.compile(r"font-style:\s*(\w+)")
_RANGE_RE = re.compile(r"unicode-range:\s*([^;]+);")

FONT_DIR.mkdir(parents=True, exist_ok=True)
css_blocks: list[str] = []
downloaded: list[tuple[str, int]] = []

for family, (slug, url) in FAMILIES.items():
    print(f"=== {family} ===")
    r = httpx.get(url, headers={"User-Agent": UA}, timeout=30.0)
    r.raise_for_status()
    faces = list(_FACE_RE.finditer(r.text))
    print(f"  CSS 里 @font-face 共 {len(faces)} 个")
    kept = 0
    for m in faces:
        if m.group("subset") != "latin":
            continue  # 只要 latin：中文走系统字体，其余子集是白费体积
        body = m.group("body")
        u = _URL_RE.search(body)
        w = _WEIGHT_RE.search(body)
        s = _STYLE_RE.search(body)
        rng = _RANGE_RE.search(body)
        if not (u and w):
            continue
        weight = w.group(1)
        style = s.group(1) if s else "normal"
        fname = f"{slug}-latin-{weight}.woff2" if style == "normal" else f"{slug}-latin-{weight}-{style}.woff2"
        target = FONT_DIR / fname
        if not target.exists():
            resp = httpx.get(u.group(1), headers={"User-Agent": UA}, timeout=30.0)
            resp.raise_for_status()
            target.write_bytes(resp.content)
        downloaded.append((fname, target.stat().st_size))
        kept += 1
        css_blocks.append(
            "@font-face {\n"
            f"  font-family: '{family}';\n"
            f"  font-style: {style};\n"
            f"  font-weight: {weight};\n"
            "  font-display: swap;\n"
            f"  src: url('/fonts/{fname}') format('woff2');\n"
            + (f"  unicode-range: {rng.group(1).strip()};\n" if rng else "")
            + "}\n"
        )
    print(f"  保留 latin 字重: {kept} 个")

header = (
    "/* T9.2 · 自托管字体（由 scripts/fetch_fonts.py 生成，勿手改）\n"
    " *\n"
    " * 为什么自托管（2026-09-15 实测修正过口径，别写错）：\n"
    " *   实测 fonts.googleapis.com 与 gstatic 的 woff2 **在本机可达**，所以\n"
    " *   **不是「加载不到」**；改自托管的真实理由是：\n"
    " *     ① 去掉首屏对境外域名的依赖（Google Fonts 即使通，也常在数百 ms～数秒量级，\n"
    " *        且墙内外网络质量差异大，属于不可控变量）；\n"
    " *     ② 离线/局域网演示可用（本项目有 `start.bat --lan` 的演示路径）；\n"
    " *     ③ 少两个 DNS/TLS 往返，首屏更稳。\n"
    " *\n"
    " * 只取 **latin 子集**：中文不走这两款字体（见 tailwind.config.js 的字体栈，\n"
    " * 中文交给 PingFang SC / Microsoft YaHei / Noto Sans SC 等系统字体），\n"
    " * 把 cyrillic/greek/vietnamese 一起下下来纯属白费体积。\n"
    " */\n"
)
CSS_OUT.write_text(header + "\n" + "\n".join(css_blocks), encoding="utf-8")

print()
print("=== 落盘 ===")
total = 0
for name, size in downloaded:
    total += size
    print(f"  {name:38s} {size / 1024:7.1f} KB")
print(f"  合计 {total / 1024:.1f} KB / {len(downloaded)} 个文件")
print(f"  CSS → {CSS_OUT}（{len(css_blocks)} 个 @font-face）")
