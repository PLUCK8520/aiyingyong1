# 案例 #17 · 默认值把"安全侧"写反了：Tavily 会静默扣额度

- **标签**：额度保护线 / 配置默认值 / 文档与实现不一致
- **发现**：2026-09-15（接 key 前的代码审查，**未上线、未造成实际扣费**）
- **状态**：已修复（`tavily_mode` 默认 `replay` + 配置项 `ATTEST_TAVILY_MODE`）

## 现象

`retrieval/tavily_client.py` 的模块文档串写着「默认 `replay`（命中本地缓存才返回）」，
但代码里是：

```python
mode: Mode = "auto"                      # ← 默认是"没缓存就联网"
```

而 `graph/build.py` 装配时**不传这个参数**：

```python
return TavilyClient(settings.tavily_api_key, DATA_DIR / "cache")   # 吃默认值 auto
```

两者相加 = **配上 `TAVILY_API_KEY` 并切 `search_mode=tavily` 后，任何没录过的 query 都会真打 API。**
一轮调研 3~5 次检索，就是 3~5 credits；而 Tavily 免费档只有 1000 credits/月。

## 根因

不是"写错了代码"，而是**默认值选在了危险侧**：

- 额度保护线的设计意图是"开发期默认不联网"，文档也这么写；
- 但 dataclass 默认值给了 `auto`（便利优先），且装配处偷懒不传参——
  于是**文档、意图、代码三者里，代码实际上站在了相反的一边**。
- 这类缺陷不会报错、不会有日志、不会有人发现，直到看账单。

## 怎么发现的

**"读文档串"和"读默认值"是两件事**。这次是把模块 docstring 的自述（默认 replay）
与 `dataclass` 字段默认值逐字对照，再顺着装配点确认没人覆盖它——三个点连起来才看见。
如果不是刻意做这个对照，代码"看起来是对的"。

## 修复

1. `mode` 默认改为 `replay`（忠于文档意图与额度保护线）；
2. 新增配置项 `ATTEST_TAVILY_MODE`（`replay`/`record`/`auto`），`build.py` **显式传入**；
3. `replay` 缓存未命中时不再静默返回空，而是打出**可执行的警告**
   （"要录这条 query：设 `ATTEST_TAVILY_MODE=record` 后重跑"）；
4. 加防回归测试两条：`test_default_mode_is_replay`（默认值断言）
   与 `test_replay_never_touches_network_when_cache_miss`（未命中时 HTTP 调用数必须为 0）。

顺带修掉同一文件里三个会在接真实检索后立刻暴露的问题：
`raw_content`（整页正文）不截断会顶爆 LLM 上下文/预算 → 加 `max_content_chars`；
重试无退避（对 429 无效）→ 加指数退避；非 JSON 响应只报裸 `JSONDecodeError`
（看不出网关返回了 HTML）→ 改为带上下文前 200 字符的可定位报错。

## 教训

1. **凡"默认值"，必须问一句"它落在安全侧还是危险侧"**。默认值是最强的隐性约定——
   用户不会读它，只会被它影响。涉及花钱、删数据、联网的开关，默认必须是**不做**那一侧。
2. **文档串与默认值要对读**。文档说 A、默认值是 B 时，没有人会收到通知；
   这次能发现，纯粹是因为做了这个逐字对照。
3. 装配点要**显式传参**。依赖"反正默认值对"是脆的——默认值一改，装配处就静默变语义。
