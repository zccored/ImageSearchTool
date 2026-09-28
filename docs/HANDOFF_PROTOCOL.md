# 跨进程交接协议（img_server → ImageSearchTool 增量建库）

> **本文件是协议的唯一权威文档**，与实现 `hybrid_search/handoff.py` 一一对应
> （字段表、校验规则、错误语义都从那里同步而来）。改协议时**两处一起改**，
> 并跑一遍 `devtools/verify_service.py` + 手工交接冒烟。
>
> 场景：img_server（下载方）完成一批下载与哈希校验后，在自己的界面点一个按钮，
> 把"这批新图落在哪"写成一份 request JSON，然后退出；本程序（ImageSearchTool）
> 由启动器拉起、读取 request、把新图**增量并进既有索引**，最后写 result JSON 回审计目录。

---

## 1. 文件流转（防重入）

```
handoff/request_<id>.json         ← img_server 写入（原子写：先 .tmp 再 os.replace）
handoff/working_<id>.json         ← 本程序接手时改名（同目录 rename，双实例抢跑只有一个成功）
handoff/result_<id>.json          ← 处理完成后写入（同样先写 .tmp 再 replace）
                                  ← working_ 在 result 写完后删除
```

- 交接目录默认 `<image-search 的上级>/handoff/`（即与本项目同级），可用环境变量
  **`IMG_HANDOFF_DIR`** 覆盖；`ingest` 命令还会用「request 文件所在目录」覆盖默认值，
  便于把交接目录放在任意位置。
- 幂等：同一批请求重复触发是安全的 —— 增量按**路径 + 内容 MD5** 去重，只有真正的新内容会入库。

## 2. request 结构（schema v1）

```jsonc
{
  "schema": 1,                                   // 必填，必须等于 1
  "kind": "download_batch_complete",             // 必填，固定值
  "request_id": "20260927-120501-abc123",        // 必填（幂等键 / result 文件名）
  "ts": "2026-09-27T12:05:01",                   // 可选，ISO 时间
  "source": "img_server",                        // 可选，来源标识
  "roots": [                                     // 必填，至少一个
    { "path": "F:\\图库\\2026-09 新番", "note": "本批 1832 张" }
  ],
  "prefix": "",                                  // 可选：留空 = 按图库根自动定位（推荐）
  "open_mode": "gui",                            // 可选："gui"（默认）| "cli"
  "expect_exit": { "pids": [], "names": [] },     // 可选，仅启动器使用（等待这些进程退出）
  "note": ""                                     // 可选
}
```

**校验规则**（不满足直接 `ValueError`，不会改任何文件）：

| 字段 | 规则 |
| :--- | :--- |
| `schema` | 必须 `== 1`，否则报"不支持的交接协议 schema" |
| `kind` | 必须 `== "download_batch_complete"` |
| `roots` | 至少一个，且每项必须有非空 `path`（无 `path` 的项被丢弃；全空则报错） |
| `open_mode` | 缺省补 `"gui"` |
| `expect_exit` | 缺省补 `{"pids": [], "names": []}` |

## 3. 图库根自动定位（`prefix` 留空的推荐用法）

下载根（`roots[].path`）**不一定是图库根**：它可能是图库根之下的某个子目录/子图集
（这种情况下该子目录里没有索引）。定位规则（`handoff.locate_gallery_root()`）：

1. 从 `path` 自身开始，逐级**向上**检查每个目录下是否存在
   `<dir>/.gallery_index/*.meta.json`；
2. 命中**最近**的一个 → 它就是图库根宿主，`prefix` 取该目录下**真实存在**的索引名
   （兼容自定义前缀，而非硬编码 `gallery`）；
3. 整条祖先链都没有 → 把 `path` 自身当作**新图库根**，用与 GUI 一致的默认位置
   `<path>/.gallery_index/gallery` 首次建库。

> 显式给了 `prefix` 则**完全尊重请求**，不做任何定位。

**非致命提示（`notices`）**：若请求根自身已有索引、而它的上级目录也存在另一套索引，
说明历史上可能在子目录里错位建过库 → 会写一条提示（不阻断处理），建议人工确认后
删掉多余的 `.gallery_index`，下次触发即自动并入上级图库索引。

## 4. result 结构

```jsonc
{
  "request_id": "…",
  "ok": true,                        // = 无 errors 且无致命异常
  "started_at": "…", "finished_at": "…",
  "prefix": "F:\\图库\\.gallery_index\\gallery",   // 第一个计划项的 prefix（审计用）
  "steps": [
    { "root": "…", "gallery_root": "…", "located": true,
      "prefix": "…", "added": 1832, "mode": "增量",   // 或 "首次构建"
      "total_in_index": 39854, "secs": 412.7 }
  ],
  "notices": ["…"],                  // 非致命提示（见第 3 节）
  "total_added": 1832,
  "total_secs": 412.7,
  "errors": []                       // 每个失败根一项 {"root": "…", "error": "repr(e)"}
}
```

- **单根失败不中断其余根**：出错的根写进 `errors[]`（并在 `steps` 里带 `error`），
  其余根照常处理；只要有一个根失败，`ok` 即为 `false`。
- 致命异常（request 读不了 / schema 不符 / 顶层异常）会写
  `{"ok": false, "fatal_error": "repr(e)"}`，并且**仍然写 result 文件**（审计闭环）。
- `added` 是"本次真正新增的索引条目数"（重复触发时为 0 属正常）。

## 5. 三个入口

| 场景 | 命令 / 方式 |
| :--- | :--- |
| **GUI 自动模式**（推荐，带进度条） | `python gui.py --auto-handoff <request.json>`；Web 版同理 `python gui_web.py`（由启动器拉起） |
| **CLI 无界面模式** | `python main.py ingest <request.json>`（返回码非 0 表示有失败根） |
| **启动器**（等 img_server 退出后再拉起） | `python handoff_launcher.py …`（读取 `expect_exit` 的 `pids`/`names`，等这些进程消失后按 `open_mode` 启动） |

无参启动 GUI/CLI 的行为与平时完全一致：交接只在**显式传入 request** 时生效，
本程序不常驻、不影响日常使用。

## 6. 不变量（img_server 侧可以依赖的保证）

1. **不改动任何图库图片**：只读扫描 + 只写 `<_图库根>/.gallery_index/` 下的索引文件；
2. **不常驻**：处理完即退出（GUI 模式下窗口由启动器决定何时关闭）；
3. **原子写**：request / result 都是"先写 `.tmp` 再 `os.replace`"，读者不会看到半个文件；
4. **幂等**：重复请求安全（路径 + MD5 去重）；
5. **失败可见**：失败根一定出现在 `result.errors[]`，致命错误一定出现在 `fatal_error`。

## 7. 排查清单

| 现象 | 先看 |
| :--- | :--- |
| 没开窗 / 没建库 | 交接目录里是 `request_*` 还是 `working_*`（后者说明接手进程中途死了，删掉 working 再重试） |
| 结果对不上 | `result.steps[].gallery_root` vs 你以为的图库根 —— 大概率是"下载根是子目录"被自动定位到上层了 |
| `added` 为 0 | 正常：这些图已在索引里（按路径 + MD5 去重）；看 `total_in_index` 是否已包含它们 |
| `schema` / `kind` 报错 | request 结构问题，对照第 2 节的字段表 |
| 卡在等待 | `expect_exit.pids` / `names` 里列出的进程还没退出（启动器在等它们） |

---

> 历史说明：早期版本 README 引用的 `docs/HANDOFF_PROTOCOL.md` 曾长期缺失（协议只存在于
> 代码注释里），2026-09-27 按 `hybrid_search/handoff.py` 的实现补回本文件；
> 那份"未随公开发布版分发"的说明随之作废。
