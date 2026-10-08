# 跨进程交接协议（img_server → ImageSearchTool 增量建库）

> **本文件是协议的唯一权威文档**，与实现 `hybrid_search/handoff.py` 一一对应
> （字段表、校验规则、错误语义都从那里同步而来）。改协议时**两处一起改**，
> 并跑一遍 `devtools/verify_service.py` + `devtools/verify_handoff_modes.py` + 手工交接冒烟。
>
> 场景：img_server（下载方）完成一批下载与哈希校验后，在自己的界面点一个按钮，
> 弹出「交接方式」对话框让用户勾选本次做哪些增量（整图 / 子图，默认两个都做），
> 把"这批新图落在哪"写成一份 request JSON，然后退出；本程序（ImageSearchTool）
> 由启动器拉起、读取 request、把新图**按选定方案增量并进既有索引**，最后写 result JSON 回审计目录。

---

## 1. 文件流转（防重入）

```
handoff/request_<id>.json         ← img_server 写入（原子写：先 .tmp 再 os.replace）
handoff/working_<id>.json         ← 本程序接手时改名（同目录 rename，双实例抢跑只有一个成功）
handoff/result_<id>.json          ← 处理完成后写入（同样先写 .tmp 再 replace）
                                  ← working_ 在 result 写完后删除
handoff/pending_<时间戳>.json     ← img_server 侧「未处理项目」：用户在对话框里一个方案
                                     都没选、点「暂不处理」或直接关窗时暂存的本次落盘信息
                                     （本程序**不读** pending_*，由 img_server 自己管理；
                                      用户可在同一对话框里点「立即交接」把它转成 request_*）
                                     字段：schema 2 / kind "download_batch_pending" /
                                     request_id "batch_<同一时间戳>" / reason
                                     ∈ {none_selected, deferred, dialog_closed} /
                                     modes / roots / request（组装好的 request 原文）
```

- 交接目录默认 `<image-search 的上级>/handoff/`（即与本项目同级），可用环境变量
  **`IMG_HANDOFF_DIR`** 覆盖；`ingest` 命令还会用「request 文件所在目录」覆盖默认值，
  便于把交接目录放在任意位置。
- 幂等：同一批请求重复触发是安全的 —— 整图增量按**路径 + 内容 MD5** 去重，
  子图增量按**路径**去重，只有真正的新内容会入库。
- **img_server 侧第二十轮起会自己打理这些文件**（本程序完全不用管）：
  ① 打开交接弹框时**读回** `result_*.json`（顶部「最近一次交接结果」摘要 + 历史下拉 + 未处理列表每行的结果标注与「查看结果」明细）；
  ② 按 TTL 30 天 / 上限 200 条自动清理最旧的 `pending_*.json`（另有「清理过期」按钮手动触发），
  并清掉崩溃残留的 `*.json.tmp`（1 天）——**只动这两类**，`request_*.json` / `result_*.json` / 图片一律不碰；
  ③ 未处理列表支持分页（每页 20 条）与多词（AND）搜索，可搜时间戳 / `request_id` / reason / modes / 根路径 / 结果状态。
  对索引器侧**没有任何新要求**：仍然只按下面的字段写 request、只写 result 即可。

## 2. request 结构（schema v2；兼容读取 v1）

```jsonc
{
  "schema": 2,                                   // 必填：2（v1 仍可读，见下）
  "kind": "download_batch_complete",             // 必填，固定值
  "request_id": "batch_20261002_153012",         // 必填（幂等键 / result 文件名）
  "ts": "2026-10-02T15:30:12",                   // 可选，ISO 时间
  "source": "img_server",                        // 可选，来源标识
  "roots": [                                     // 必填，至少一个
    { "path": "F:\\图库\\2026-09 新番", "note": "本批 1832 张" }
  ],
  "modes": ["full", "tiles"],                    // 可选：本次做哪些增量（见第 3 节）
  "prefix": "",                                  // 可选：留空 = 按图库根自动定位（推荐）
  "open_mode": "gui",                            // 可选："gui"（默认）| "cli"
  "expect_exit": { "pids": [], "names": [] },     // 可选，仅启动器使用（等待这些进程退出）
  "note": ""                                     // 可选
}
```

**校验规则**（不满足直接 `ValueError`，不会改任何文件）：

| 字段 | 规则 |
| :--- | :--- |
| `schema` | 必须是 `2` 或 `1`（`SUPPORTED_SCHEMAS`）；其它值报"不支持的交接协议 schema" |
| `kind` | 必须 `== "download_batch_complete"` |
| `roots` | 至少一个，且每项必须有非空 `path`（无 `path` 的项被丢弃；全空则报错） |
| `modes` | 可选数组，取值只能是 `"full"` / `"tiles"`（含 `整图`/`子图`/`瓦片` 等别名）；缺省、`null`、空数组都视同 `["full", "tiles"]`；未知取值直接报错 |
| `open_mode` | 缺省补 `"gui"` |
| `expect_exit` | 缺省补 `{"pids": [], "names": []}` |

> **v1 → v2 兼容**：schema v1 的请求没有 `modes` 字段，一律按"两个方案都做一遍"
> 处理（与 v1 时代的行为等价）；v1 请求不需要任何改动就能继续用。

## 3. 交接方案（`modes`：整图 / 子图）

`modes` 决定本次交接做哪些增量，**执行顺序固定：先 `full`（整图），后 `tiles`（子图）**。
规格化由 `handoff.normalize_modes()` 完成（去重 + 固定顺序 + 别名识别）。

| 取值 | 含义 | 落盘位置 | 增量语义 |
| :--- | :--- | :--- | :--- |
| `"full"` | 整图（粗筛 + ResNet 精排）增量入库 | `<图库根>/.gallery_index/<前缀>.*` | 复用 `HybridEngine.add`：路径 + MD5 去重；索引不存在时首次从零构建 |
| `"tiles"` | 子图（512px 瓦片，overlap 0.25）增量入库 | 与整图索引**同目录**的 `gallery_tiles.*`（`tile_index.tiles_prefix_of(<前缀>)`） | 已有瓦片索引 → `tile_index.add_tiles`（按路径去重）；没有 → `build_tiles` 首次切块构建 |

- **默认（用户两个都勾）**：先整图、后子图，同一批新图各走一遍；两个阶段互不牵连 ——
  整图阶段失败不影响子图阶段（反之亦然），失败项分别记入 `errors[]`（带 `"mode"` 字段）。
- **只勾一个**：只做该方案；`modes` 就只含那一个值。
- **子图增量依赖整图索引吗？** 不依赖。瓦片索引与整图索引各自独立存放，只有"同目录"这一层
  关系；但首次为某个图库建瓦片索引会**重新扫描并切块该图库下的全部图片**（耗时较长，属正常）。
- **一个方案都没选**：不会产生 request 文件 —— img_server 侧把本次落盘信息写进
  `handoff/pending_<时间戳>.json` 并列入「未处理项目列表」，由用户以后点「立即交接」补做。

## 4. 图库根自动定位（`prefix` 留空的推荐用法）

下载根（`roots[].path`）**不一定是图库根**：它可能是图库根之下的某个子目录/子图集
（这种情况下该子目录里没有索引）。定位规则（`handoff.locate_gallery_root()`）：

1. 从 `path` 自身开始，逐级**向上**检查每个目录下是否存在
   `<dir>/.gallery_index/*.meta.json`；
2. 命中**最近**的一个 → 它就是图库根宿主，`prefix` 取该目录下**真实存在**的索引名
   （兼容自定义前缀，而非硬编码 `gallery`）；
3. 整条祖先链都没有 → 把 `path` 自身当作**新图库根**，用与 GUI 一致的默认位置
   `<path>/.gallery_index/gallery` 首次建库（子图索引随之落在 `<path>/.gallery_index/gallery_tiles`）。

> 显式给了 `prefix` 则**完全尊重请求**，不做任何定位。

**非致命提示（`notices`）**：若请求根自身已有索引、而它的上级目录也存在另一套索引，
说明历史上可能在子目录里错位建过库 → 会写一条提示（不阻断处理），建议人工确认后
删掉多余的 `.gallery_index`，下次触发即自动并入上级图库索引。
首次为某个根构建子图索引时，也会记一条提示（说明"本次是首次切块、耗时长"）。

## 5. result 结构

> img_server 侧第二十轮起会**回读**这个文件（交接弹框顶部摘要 + 历史下拉 + 未处理列表每行的结果标注与
> 「查看结果」明细），判定只用 `ok` / `errors` / `fatal_error` / `total_added` / `total_tiles_added` /
> `steps[].stages[]`；损坏的 JSON 会被显示为「结果文件损坏」而不会让弹框打不开。

```jsonc
{
  "request_id": "batch_20261002_153012",
  "ok": true,                        // = 无 errors 且无致命异常
  "started_at": "…", "finished_at": "…",
  "prefix": "F:\\图库\\.gallery_index\\gallery",   // 第一个计划项的 prefix（审计用）
  "modes": ["full", "tiles"],        // 本次实际执行的方案（顺序即执行顺序）
  "steps": [
    { "root": "…", "gallery_root": "…", "located": true,
      "prefix": "…",
      "added": 1832, "mode": "增量", "total_in_index": 39854,   // 兼容字段：整图阶段汇总
      "tiles_added": 0, "tiles_mode": "增量",
      "tiles_prefix": "F:\\图库\\.gallery_index\\gallery_tiles",
      "tiles_secs": 21.4, "secs": 434.1,                        // secs = 该根两阶段合计
      "stages": [                                               // 逐方案明细（顺序 = 执行顺序）
        { "mode": "full",  "label": "整图增量",      "prefix": "…\\gallery",
          "added": 1832, "build_mode": "增量", "total_in_index": 39854, "secs": 412.7 },
        { "mode": "tiles", "label": "子图(瓦片)增量", "prefix": "…\\gallery_tiles",
          "added": 0,    "build_mode": "增量", "total_tiles": 39854,    "secs": 21.4 }
      ] }
  ],
  "notices": ["…"],                  // 非致命提示（见第 4 节）
  "total_added": 1832,               // 整图新增条目合计
  "total_tiles_added": 0,            // 子图新增瓦片合计
  "total_secs": 434.1,               // 两个阶段合计
  "errors": []                       // 每个失败项一项 {"root": "…", "mode": "full|tiles", "error": "repr(e)"}
}
```

- **单根 / 单方案失败不中断其余**：出错的项写进 `errors[]`（`mode` 标明是整图还是子图阶段，
  并在对应 `stages[]` 项里带 `error`），其余照常处理；只要有一个失败项，`ok` 即为 `false`。
- 致命异常（request 读不了 / schema 不符 / 顶层异常）会写
  `{"ok": false, "fatal_error": "repr(e)"}`，并且**仍然写 result 文件**（审计闭环）。
- `added` / `tiles_added` 是"本次真正新增的条目数"（重复触发时为 0 属正常）；
  `build_mode` 为 `"首次构建"` 或 `"增量"`。
- 老版本本程序读到带 `modes` 的 v2 请求会直接报 schema 不支持 —— 升级本程序后再用。

## 6. 三个入口

| 场景 | 命令 / 方式 |
| :--- | :--- |
| **GUI 自动模式**（推荐，带进度条） | `python gui.py --auto-handoff <request.json>`（`--auto-handoff` 只存在于桌面 GUI；Web 版 `gui_web.py` 目前没有交接入口 —— 它由人工在网页上建库/检索） |
| **CLI 无界面模式** | `python main.py ingest <request.json>`（返回码非 0 表示有失败项）；可用 `--modes full` / `--modes tiles` / `--modes full,tiles` 覆盖请求里的方案 |
| **启动器**（等 img_server 退出后再拉起） | `python handoff_launcher.py <request.json> [--wait 90]`（读取 `expect_exit` 的 `pids`/`names`，等这些进程消失后按 `open_mode` 启动）；cli 模式跑完打印 `[launcher] ingest ok=True 方案 full+tiles 新增 N 张 / M 瓦片`，返回码同 `ingest` |

无参启动 GUI/CLI 的行为与平时完全一致：交接只在**显式传入 request** 时生效，
本程序不常驻、不影响日常使用。GUI 自动模式在两段之间会切换进度条阶段名
（`fused` → `tiles`），不会在整图做完时就显示"全部完成"。

## 7. 不变量（img_server 侧可以依赖的保证）

1. **不改动任何图库图片**：只读扫描 + 只写 `<_图库根>/.gallery_index/` 下的索引文件；
2. **不常驻**：处理完即退出（GUI 模式下窗口由启动器决定何时关闭）；
3. **原子写**：request / result 都是"先写 `.tmp` 再 `os.replace`"，读者不会看到半个文件；
4. **幂等**：重复请求安全（整图路径 + MD5 去重、子图路径去重）；
5. **失败可见**：失败项一定出现在 `result.errors[]`（带 `mode`），致命错误一定出现在 `fatal_error`；
6. **只认 `request_*` / `working_*`**：`pending_*` 是 img_server 自己的未处理清单，本程序不读、不写；
7. **result 与 request 同目录**：无论正常跑完还是"等 img_server 退出"超时，`result_<request_id>.json`
   一律写在 **request 文件所在目录**（即 img_server 的 `handoff\`），不会落到本程序自己的
   `image-search\handoff\` 里 —— img_server 侧始终能看到这条记录。

## 8. 排查清单

| 现象 | 先看 |
| :--- | :--- |
| 没开窗 / 没建库 | 交接目录里是 `request_*` 还是 `working_*`（后者说明接手进程中途死了，删掉 working 再重试） |
| 结果对不上 | `result.steps[].gallery_root` vs 你以为的图库根 —— 大概率是"下载根是子目录"被自动定位到上层了 |
| `added` 为 0 | 正常：这些图已在索引里（按路径 + MD5 去重）；看 `total_in_index` 是否已包含它们 |
| 只做了整图、没做子图 | 看 `result.modes` 与 `steps[].stages`：多半是 img_server 弹框里用户只勾了「整图增量入库」 |
| 子图阶段很慢 | 首次为该图库建瓦片索引要全量切块（`build_mode: "首次构建"`），之后同一批重跑是增量、只处理新图 |
| `schema` / `kind` / `modes` 报错 | request 结构问题，对照第 2 节的字段表 |
| 卡在等待 | `expect_exit.pids` / `names` 里列出的进程还没退出（启动器在等它们）；超过 `--wait`（默认 90s）会放弃并在**请求所在目录**写一条 `ok:false` + `fatal_error: 等待 img_server 退出超时…` 的 result，图库不动 |

---

> 历史说明：早期版本 README 引用的 `docs/HANDOFF_PROTOCOL.md` 曾长期缺失（协议只存在于
> 代码注释里），2026-09-27 按 `hybrid_search/handoff.py` 的实现补回本文件；
> 那份"未随公开发布版分发"的说明随之作废。
> 2026-10-02：协议升到 **schema v2**，新增 `modes`（整图 / 子图选做或都做）与
> `steps[].stages` 明细；v1 请求继续可用（等同"两个都做"）。
