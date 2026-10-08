# img_server（全栈图库管理器）侧接入需求 · 交接协议 schema v2

> 面向对象：维护 `全栈图库管理器 v3.2bata\img_server.py` 的开发者 / 智能体。
> 本文件讲**对方（img_server）侧的交接接口**；索引器侧的契约全文见
> `image-search/docs/HANDOFF_PROTOCOL.md`（那份是权威版本，两文件冲突时以它为准）。
> 版本：2026-10-02 · schema v2（整图 / 子图 可选或都做）+ 第二十轮（result 回读 / pending 清理 / 列表分页搜索）。
>
> **实现状态：img_server 侧已实施**（弹框 / pending 暂存 / 未处理列表第一轮落地；
> 第二十轮又加了 **result 回读**、**pending 清理策略**、**列表分页 + 搜索**，自检见 §8）。
> §1 的行号是**第二十轮改后**版本（`img_server.py` **16297 行 / 762477 B**）的落点
> （第一轮改后是 15693 行 / 733808 B；协议段内新增函数把其后所有行号整体下移）；
> 本文档剩余部分（§3 请求字段、§5 定位、§7 回读）是索引器侧的契约说明，双方共用。

## 0. 一句话

img_server 的「🔁 下载完成→图库检索」按钮，现在要在**用户勾选交接方案**之后再发起
交接：勾了什么就写进请求文件的 `modes`，索引器按 `full → tiles` 顺序连做；
用户什么都没勾（或选了"暂不处理"、直接关窗）时**不发起交接**，改为把本批落盘信息
暂存成 `handoff\pending_<时间戳>.json`，并在他自己的「未处理项目」列表里提供
**立即交接**与**删除**。

## 1. 实现落点（行号 = 2026-10-02 改后版本）

交接链路（改动部分加粗）：

| 位置 | 作用 |
| :--- | :--- |
| `9115` 附近 | `self.btn_handoff = QPushButton("🔁 下载完成→图库检索")` |
| `13504-13537` `_on_handoff_clicked()` | 前置校验（原样保留）→ **弹 `HandoffModeDialog(...).exec()`** 按勾选发起或暂存 |
| `13539-13562` `_quit_after_handoff()` | 交接后退出主程序（过渡进程靠 psutil 轮询确认它已退出） |
| `15110` / `15111` | `HANDOFF_MODES = ('full','tiles')`、`HANDOFF_MODE_LABELS` |
| `15116` / `15117` | `HANDOFF_PENDING_REASONS = ('none_selected','deferred','dialog_closed')` 及中文标签 |
| `15123-15126` | **第二十轮**：`HANDOFF_PENDING_TTL_DAYS = 30`、`HANDOFF_PENDING_MAX_RECORDS = 200`、`HANDOFF_PENDING_PAGE_SIZE = 20`、`HANDOFF_TMP_TTL_DAYS = 1` |
| `15129` `_normalize_handoff_modes(modes)` | 过滤非法值 + 去重 + 固定顺序（`full` 在前）；`None` = 两个都做 |
| `15147` `_handoff_now_stamp()` | `time.strftime('%Y%m%d_%H%M%S')`（request_id / pending 文件名同源） |
| `15147` `_free_handoff_stamp(prefix, dirpath=None, want_ts=None, ext='.json')` | 取不撞文件的 `<prefix>_<时间戳><ext>`（同秒连点向后顺延，避免互相覆盖） |
| `15168` `_stamp_to_iso(stamp)` | `20261002_153012` → `2026-10-02T15:30:12`（文件名与内容时间同源） |
| `15176` `_build_handoff_request(roots, extra_pids=(), note='', open_mode='gui', modes=None, request_id=None)` | 组装 request 字典（v2：多了 `modes`） |
| `15204` `_write_handoff_request_json(...)` / `15217` `_write_handoff_request_file(..., dirpath=None, modes=None, request_id=None)` | `.tmp` + `os.replace` 原子写 `request_<id>.json` |
| `15229` `_write_pending_record(...)` | 写 `pending_<时间戳>.json`（非法 `reason` 直接 `ValueError`） |
| `15267` `_list_pending_records(dirpath)` | 读 `handoff\pending_*.json`，**时间倒序** |
| `15294` `_delete_pending_file(...)` / `15305` `_write_handoff_request_from_record(...)` | 红 × 删除 / 「立即交接」按暂存记录重写 request |
| `15351` / `15378` / `15386` / `15411` | **第二十轮**：`_pending_record_epoch` / `_pending_record_age_days` / `_cleanup_handoff_tmp_files` / `_cleanup_pending_records`（只删 `pending_*.json` 与 `*.json.tmp`） |
| `15457` / `15465` / `15497` | **第二十轮**：`_result_file_for` / `_load_handoff_result`（带 `(fp,mtime,size)` 缓存）/ `_list_handoff_results` |
| `15524` / `15530` / `15563` / `15626` | **第二十轮**：`_short_text` / `_summarize_handoff_result`（六档状态口径）/ `_format_handoff_result_detail` / `_pending_search_haystack` |
| `15648` `_launch_handoff_launcher(req_path, wait=90)` | `DETACHED_PROCESS` 起 `[python, -X, utf8, HANDOFF_LAUNCHER, req_path, --wait, str(wait)]` |
| `15679-16297` `class HandoffModeDialog(QDialog)` | 见 §2（`__init__` 15708、`_build_ui` 15735、`_roots_summary` 15882、`reload_results` 15893、`_selected_result` 15915、`_on_result_combo_changed` 15928、`_result_for` 15942、`_on_view_recent` 15952、`_show_result_detail` 15960、`_on_view_result` 15966、`reload_pending` 15981、`_apply_filter` 15998、`_on_search_changed` 16024、`_on_page_prev` 16028、`_on_page_next` 16033、`_render_page` 16039、`_pending_row_text` 16093、`_pending_row_tooltip` 16105、`_on_cleanup_clicked` 16117、`_show_cleanup_notice` 16142、`selected_modes` 16159、`_stash(reason)` 16168、`_launcher_ready` 16176、`_on_start_clicked` 16186、`_on_defer_clicked` 16218、`_on_handoff_record` 16235、`_on_delete_record` 16255、`_quit_via_host` 16269、`_stash_on_close` 16281、`closeEvent` 16290、`reject` 16294） |

> `15110-15121` 里原有的 `HANDOFF_DIR = <管理器目录>\handoff` 与
> `HANDOFF_LAUNCHER = <管理器目录>\image-search\handoff_launcher.py` 未变，
> 只是被上面这些新函数复用。

## 2. 交接方案弹框（已实现）

`_on_handoff_clicked()` 里的确认框换成 `HandoffModeDialog`（`QDialog`），内容：

1. **两个复选框**（默认**都勾选**；`selected_modes()` 16159 汇总结果）：
   - ☑ 整图增量（`full`）—— 新图并入整图索引 `gallery.*`
   - ☑ 子图(瓦片)增量（`tiles`）—— 新图切 512px 瓦片并入 `gallery_tiles.*`
   - 界面上带提示：首次没有子图索引时会自动切块构建，**较慢**（本机实测约 434 块/秒，
     1 万张图 ≈ 约 13 万块 ≈ 5 分钟量级）。
2. **三个按钮**：
   - `开始交接`（`_on_start_clicked` 16186）：至少勾了一项 → 按 §3 写 request 并
     `_launch_handoff_launcher(fp, wait=90)` → `_quit_via_host()` 16269 退出主程序；
     **一项都没勾 → 等同"暂不处理"**（`_stash('none_selected')`，不启动也不退出）。
   - `暂不处理`（`_on_defer_clicked` 16218）→ `_stash('deferred')`。
   - `关闭`（右上角 × / Esc，`closeEvent` 16290 / `reject` 16294）→ `_stash('dialog_closed')`；
     若本次已有落盘信息（`roots` 非空）即暂存（Esc 同样暂存）。
3. **未处理项目列表**（`reload_pending` 15981 刷新、`_pending_row_text` 16093 渲染、
   `_list_pending_records` 15267 按时间倒序）：
   - 表名 = 文件名里的时间戳（`pending_20261002_153012.json` → `2026-10-02 15:30:12`），附 roots 摘要；
   - 每行最右侧两个操作：
     - **`立即交接`**（`_on_handoff_record` 16235）：由 `_write_handoff_request_from_record` 15305
       取出暂存里的 request 原文 → 补上当前勾选与**新的** `request_id`/`ts` → 写
       `request_<新id>.json` → 启动过渡进程 → 删掉该 pending → 退出主程序；
     - **红 ×`删除`**（`_on_delete_record` 16255 → `_delete_pending_file` 15294）：
       确认一次后删除该 pending（删除即视为"已处理"）。
   - 列表为空时显示灰字提示"暂无未处理项目"。

### 2.1 pending 暂存文件（img_server 私有）

- 路径：`<管理器目录>\handoff\pending_<YYYYmmdd_HHMMSS>.json`（**一个批次一个文件**）。
  时间戳 `time.strftime('%Y%m%d_%H%M%S')`，同秒撞车由 `_free_handoff_stamp` 15147
  顺延下一秒；文件内 `request_id == "batch_<同一时间戳>"`。
- 写盘方式与 request 一致：先写 `.tmp` 再 `os.replace`（原子）。
- `reason` 只接受 `none_selected` / `deferred` / `dialog_closed`，其它取值
  `_write_pending_record` 15229 直接抛 `ValueError`。
- **索引器侧不读不写 pending 文件**（`list_requests()` 只列 `request_*.json`），
  它属于 img_server 的私有待办台账。
- **清理策略（第二十轮，用户在管理器进度里能看到）**：打开弹框时按
  `HANDOFF_PENDING_TTL_DAYS = 30` 天 / `HANDOFF_PENDING_MAX_RECORDS = 200` 条自动清理最旧的
  （另有「清理过期」按钮手动触发）；只删 `pending_*.json` 与崩溃残留的 `*.json.tmp`（后者 1 天），
  **不碰** `request_*.json`、`result_*.json` 与图片。索引器完全不受影响。

### 2.2 第二十轮新增：result 回读 / 清理 / 分页搜索

对着索引器写出的 `result_<request_id>.json`，管理器现在会**读回来给人看**（不自动重试、不弹通知）：

- **弹框顶部**「交接结果」分组：最近一次结果摘要 `最近一次交接结果（<finished_at>）：✅ 新增 N 张 / M 瓦片（X.Xs）`
  + 历史下拉 + 「查看详情」（`_show_result_detail` 15960，多行明细含逐根目录逐阶段）。
- **未处理列表每行**「结果」列标注该批次的状态（六档：`✅ 全成功` / `⚠ 部分成功` / `❌ 失败：<errors[0].error>` /
  `❌ 失败：<fatal_error>` / `❌ 结果文件损坏` / `－ 无结果（未执行或结果未落盘）`），行内「查看结果」按钮弹明细。
- 结果只从**弹框自己的目录**（缺省 `<管理器目录>\handoff`）读，按 `(文件, mtime, 大小)` 缓存。
- **列表分页**：每页 20 条（`HANDOFF_PENDING_PAGE_SIZE`），底部 `共 N 条 ｜ 第 p/q 页` + 上/下页。
- **搜索**：一个输入框，空格分隔多词 = 全部命中（AND）；可搜时间戳 / `request_id` / `reason`（原值或中文）/
  `modes`（原值或中文）/ 根目录路径 / 结果状态（含「无结果」「失败」「✅」「⚠」），每次筛选回到第 1 页。
- 表格因此从第一轮的三列变成五列：`批次信息 / 结果 / 查看结果 / 立即交接 / 删除`。

> 对索引器侧**没有任何新要求**：仍然只写 `result_<request_id>.json`（§7 的字段），管理器自己解析；
> `ok` / `total_added` / `total_tiles_added` / `errors` / `fatal_error` 就是它读的那几个键。

## 3. request 文件（schema v2）

写到 `<管理器目录>\handoff\request_<request_id>.json`（UTF-8，缩进随意）：

```json
{
  "schema": 2,
  "kind": "download_batch_complete",
  "request_id": "batch_20261002_153012",
  "ts": "2026-10-02T15:30:12",
  "source": "img_server",
  "modes": ["full", "tiles"],
  "roots": [
    {"path": "F:/视频/Fenriruu-riru", "note": "本批 1065 张"}
  ],
  "prefix": "",
  "open_mode": "gui",
  "expect_exit": {"pids": [12345], "names": ["python.exe", "pythonw.exe"]},
  "note": ""
}
```

字段说明：

| 字段 | 必填 | 说明 |
| :--- | :--- | :--- |
| `schema` | ✅ | **2**（索引器同时兼容 1，但 1 没有 `modes`，等价于两个都做） |
| `kind` | ✅ | 固定 `"download_batch_complete"` |
| `request_id` | ✅ | `batch_<时间戳>`；result 文件名由它决定（`result_<request_id>.json`） |
| `ts` | 建议 | `%Y-%m-%dT%H:%M:%S` |
| `source` | 建议 | `"img_server"` |
| `modes` | ✅（v2） | `["full","tiles"]` / `["full"]` / `["tiles"]`；**顺序无关**，索引器固定按 `full → tiles` 执行；缺省 = 两个都做 |
| `roots[]` | ✅ | 每个 `{"path": <绝对路径>, "note": <字符串>}`；至少一个，路径必须存在 |
| `prefix` | 可选 | 留空 `""` = **自动定位图库根**（推荐，见 §5）；显式给出则所有 roots 都并入该前缀 |
| `open_mode` | 可选 | `"gui"`（默认，打开图形界面跑）/ `"cli"`（无界面直接跑完） |
| `expect_exit` | 可选 | `{"pids": [...], "names": [...]}`：过渡进程要等的进程；不填则不等 |
| `note` | 可选 | 自由文本，会出现在索引器 result 里 |

## 4. pending 文件（schema v2 · img_server 私有）

```json
{
  "schema": 2,
  "kind": "download_batch_pending",
  "request_id": "batch_20261002_153012",
  "ts": "2026-10-02T15:30:12",
  "source": "img_server",
  "reason": "none_selected",
  "modes": [],
  "roots": [{"path": "F:/视频/Fenriruu-riru", "note": "本批 1065 张"}],
  "expect_exit": {"pids": [12345], "names": []},
  "request": { "...按 §3 组装好的完整 request，未写盘..." }
}
```

| 字段 | 说明 |
| :--- | :--- |
| `kind` | 固定 `"download_batch_pending"`（schema `2`、`source` `"img_server"`） |
| `reason` | `"none_selected"`（一项都没勾却点了「开始交接」）/ `"deferred"`（点「暂不处理」）/ `"dialog_closed"`（× / Esc / 「关闭」）；其它取值写盘时抛 `ValueError` |
| `modes` | 当时勾了什么（都没勾就是 `[]`） |
| `request` | **组装好的完整 request 原文**（含 `schema`/`kind`/`roots`/`prefix`/`expect_exit` 等）。"立即交接"时把它取出来、补上当前 `modes` 与新的 `request_id`/`ts`，写成 `request_*.json` 即可 —— 这样暂存与立即交接的字段不会漂移 |

## 5. 图库根自动定位（对方不用管，但要知道）

`roots[].path` 允许是图库根下面的**子目录**（例如新下载批落在
`F:\视频\Fenriruu-riru\2026-10-02`）：`prefix` 留空时索引器会沿祖先目录向上找
最近一个含 `.gallery_index` 的目录当图库根，把新图**并入**既有索引，
不会在子目录里另建一套。判断结果会写进 result 的
`steps[].gallery_root` / `steps[].located`。

## 6. 启动过渡进程（不变）

```python
subprocess.Popen([sys.executable, "-X", "utf8", HANDOFF_LAUNCHER, req_path,
                  "--wait", "90"], creationflags=DETACHED_PROCESS, close_fds=True)
```

- 过渡进程会先等 `expect_exit` 里的进程退出（最长 90 s），再按 `open_mode`
  打开索引器 GUI（`python gui.py --auto-handoff <working_request.json>`）或直接跑完。
- 所以 img_server 写完 request 之后**必须退出主程序**（现有
  `_quit_after_handoff()` 行为不变），否则过渡进程会一直等到超时。
- 「立即交接」同样走这条链路。

## 7. result 回读（已实现：管理器弹框侧）

索引器完成后写 `<管理器目录>\handoff\result_<request_id>.json`（schema v2）：

```json
{
  "request_id": "batch_20261002_153012",
  "ok": true,
  "modes": ["full", "tiles"],
  "started_at": "...", "finished_at": "...",
  "prefix": "F:\\视频\\.gallery_index\\gallery",
  "total_added": 1065, "total_tiles_added": 13845, "total_secs": 412.3,
  "steps": [{
    "root": "F:/视频/Fenriruu-riru", "gallery_root": "F:\\视频", "located": true,
    "prefix": "F:\\视频\\.gallery_index\\gallery",
    "modes": ["full", "tiles"], "added": 1065, "mode": "增量",
    "total_in_index": 38911, "tiles_added": 13845, "tiles_mode": "增量",
    "tiles_prefix": "F:\\视频\\.gallery_index\\gallery_tiles", "tiles_secs": 294.3,
    "secs": 412.3,
    "stages": [
      {"mode": "full", "label": "整图增量", "prefix": "…\\gallery",
       "added": 1065, "build_mode": "增量", "total_in_index": 38911, "secs": 118.0},
      {"mode": "tiles", "label": "子图(瓦片)增量", "prefix": "…\\gallery_tiles",
       "added": 13845, "build_mode": "增量", "total_tiles": 13845, "secs": 294.3}
    ]
  }],
  "notices": [], "errors": []
}
```

> `stages[]` 里两个阶段都用 **`added`** 报"本次新增数"（整图=张数、子图=瓦片块数），
> `build_mode` 为 `"首次构建"` / `"增量"`；step 上的 `added`/`tiles_added` 是便捷汇总。

img_server 侧**第二十轮已经把这些字段读回界面**（见 §2.2）：弹框顶部给最近一次结果摘要 + 历史下拉 +
「查看详情」，未处理列表每行给该批次的结果标注与「查看结果」；判定用的是 `ok` / `errors` / `fatal_error` /
`total_added` / `total_tiles_added` / `steps[].stages[]`（整图段读 `total_in_index`、瓦片段读 `total_tiles`）。
`errors[]` 项形如 `{"root": ..., "mode": "full"|"tiles", "error": "..."}`。
仍**没有**自动重试、没有系统通知 —— 只是"让人看得见"。

## 8. 验收清单（已自测通过）

1. 两个都勾 → request 里 `"modes": ["full","tiles"]`，索引器 result 两个阶段都有产出。
2. 只勾整图 → `"modes": ["full"]`，只动 `gallery.*`；只勾子图 → 只动 `gallery_tiles.*`。
3. 一项都不勾点「开始交接」/ 点「暂不处理」/ 直接关窗 → **不启动过渡进程**，
   `handoff\pending_<时间戳>.json` 落盘且字段齐全（`reason` 对应三种情况）。
4. 未处理列表：表名是时间戳；`立即交接`能写出新 `request_*.json` 并启动、随后删掉
   pending；红 × 能删除单个 pending；列表为空时显示占位提示。
5. 关闭管理器后 pending 仍在，重启管理器后列表能读回来。
6. **（第二十轮）result 回读**：交接完成后重开弹框，顶部显示最近一次结果摘要；未处理列表对应行的「结果」列显示
   `✅ 新增 N 张 / M 瓦片` 或失败原因；「查看结果」弹出多行明细；没有 result 的行显示 `－ 无结果（未执行或结果未落盘）`
   且按钮置灰。
7. **（第二十轮）pending 清理**：造 30 天前的 `pending_*.json` 或超过 200 条 → 打开弹框时自动清理最旧的并在弹框里
   提示清了几条；「清理过期」按钮可手动触发；`request_*.json` / `result_*.json` / 图片**一个都没少**。
8. **（第二十轮）分页 + 搜索**：pending 超过 20 条时分页显示、上/下页可用；搜索框输入多个关键词（空格分隔）
   只留下全部命中的行，能按「无结果」「失败」这类结果文案筛出来；清空搜索恢复全部并回到第 1 页。

### 自检脚本与结果（2026-10-02）

| 脚本 | 位置 | 结果 |
| :--- | :--- | :--- |
| `verify_handoff_mode_dialog.py` | `<管理器目录>\HANDOFF-DSH\` | **ALL PASS（141 项）**，exit 0（offscreen Qt、monkeypatch 掉真实启动/退出）。第一轮 95 项 + 第二十轮 §11 结果回读 22 项 / §12 清理策略 11 项 / §13 分页+搜索 13 项 |
| `verify_handoff_indexer_crosscheck.py` | `<管理器目录>\HANDOFF-DSH\` | **ALL PASS（18 项）**，exit 0：img_server 写 → 索引器 `hybrid_search.handoff.validate()` 读，5 种 modes 情形一致；`list_requests()` 不列 `pending_*` |
| `verify_handoff_modes.py` | `image-search\devtools\` | **ALL PASS（80 项）**：modes 归一化 / 只整图 / 只子图 / 两个都做 / 幂等 / 定位 / 服务层事件流 / `handoff_launcher.py` cli 模式端到端 / 启动器等待超时分支（错误 result 落在 request 所在目录） |

备份与哈希（`<管理器目录>\HANDOFF-DSH\backup\`）：

| 版本 | 文件 | 大小 / 行数 | SHA256 |
| :--- | :--- | :--- | :--- |
| 第一轮改前（09-07 基线 + 老确认框） | `img_server.py.bak_20261002_145513` | 709535 B / 15186 行 | `1B8F11FCF5E67F3473D9CE43B106B1AC57233BD8D55162715CCBF373262D0AB5` |
| 第一轮改后 / 第二十轮改前 | `img_server.py.bak_20261002_151548` | 733808 B / 15693 行 | `9091B5C6DF568F1BB1D255FB89D0F4C4C23D245C758A0641171670574A2741EE` |
| **第二十轮改后（当前）** | `img_server.py`（本体） | **762477 B / 16297 行** | `0BA6A7C6DF3923A0F26BC55EE678F222C4B991EF8B448E25928B75A4DC497BB4` |

> 回滚第二十轮（保留弹框、去掉回读/清理/分页）：
> `copy HANDOFF-DSH\backup\img_server.py.bak_20261002_151548 img_server.py`。
