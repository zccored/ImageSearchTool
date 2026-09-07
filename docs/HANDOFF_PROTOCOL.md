# img_server → 图库检索管理器：自动交接协议（schema v1）

目标场景：img_server 完成“新下载 + 内容哈希校验”后，用户在按钮上确认 →
img_server 及上级主程序退出 → 自动打开图库检索管理器（image-search/gui.py）
→ 自动增量建库（校验 + 去重）。全程不需要人工传路径。

本文档是**双方共同遵守的进程间契约**。img_server 侧只负责“写 request + 退出 +
启动过渡进程”；接收/校验/建库/结果回写全部由 image-search 侧完成。

---

## 1. 目录约定

交接目录（双方共享、可配置）：

```
D:\code\新的代码\全栈图库管理器 v3.2bata\handoff\
    request_<id>.json      写入方（img_server）产出，待处理
    working_<id>.json      接收方开始处理后改名（防重入）
    result_<id>.json       接收方完成后的结果（含每批新增数/耗时/错误）
```
> 目录不存在时自动创建。文件流转全部使用原子操作（写 .tmp 再 os.replace）。
> 若放在其它位置：img_server 侧写 request 时自行保证目录可写；
> 命令行处理方式与目录无关（见第 5 节）。

## 2. request JSON（img_server 按钮回调写入）

```json
{
  "schema": 1,
  "kind": "download_batch_complete",
  "request_id": "batch_20260906_2350",
  "ts": "2026-09-06T23:50:00",
  "source": "img_server",
  "roots": [
    {"path": "D:\\下载内容根目录", "note": "本次批次新增/变更的图片根"}
  ],
  "prefix": "",
  "open_mode": "gui",
  "expect_exit": {
    "pids": [12345],
    "names": ["图库主程序.exe", "img_server.exe"]
  },
  "note": ""
}
```

字段说明：
| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `schema` | 是 | 恒为 `1` |
| `kind` | 是 | 恒为 `download_batch_complete` |
| `request_id` | 是 | 唯一串（建议时间戳+序号） |
| `roots` | 是 | ≥1 个目录；接收方会对每个目录递归增量入库（路径+MD5 去重，重复内容安全跳过） |
| `prefix` | 否 | 图库检索索引前缀；**留空 = 自动定位图库根**（见 4.5 节），而不是简单拼在第一个 root 下 |
| `open_mode` | 否 | `gui`（默认，打开图库检索管理器并自动执行）/ `cli`（无窗口执行） |
| `expect_exit` | 否 | launcher 等待退出的进程；`names` 用具体 exe 名，**禁止写 python.exe/普遍进程名** |

## 3. 时序

```
[img_server]  下载完成 + 哈希校验完成
    │ 1) 写 request_<id>.json（唯一 id；重复写同 id 无害，接收方幂等）
    │ 2) 启动过渡进程（一次性的，detached）：
    │      python "D:\code\新的代码\全栈图库管理器 v3.2bata\image-search\handoff_launcher.py"
    │            "…\handoff\request_<id>.json" --wait 90
    │ 3) 安全退出 img_server 及上级主程序（释放 80%+ 内存）
    ▼
[handoff_launcher.py]（轻量，等待期占用极小）
    │ 轮询 expect_exit 全部退出（超时则写失败 result，不强制结束任何进程）
    ▼ open_mode=gui
[gui.py --auto-handoff <working 文件>]   ← 图库检索管理器
    │ 自动：读取 request → 校验 → 对每个 root 增量建库（首次自动从零构建）
    │ 全程界面可见（进度条/日志/状态栏）；完成后：
    ▼ 写 result_<id>.json，删除 working_<id>.json
[img_server / 用户]  读取 result 审计（每 root 新增张数、耗时、错误）
```

## 4. result JSON（接收方写回）

```json
{
  "request_id": "batch_…",
  "ok": true,
  "started_at": "2026-09-06T23:51:00",
  "finished_at": "2026-09-06T23:51:32",
  "prefix": "D:\\…\\.gallery_index\\gallery",
  "total_added": 1234,
  "total_secs": 31.8,
  "notices": [],
  "steps": [
    {"root": "D:\\…子目录", "gallery_root": "D:\\图库根",
     "located": true, "prefix": "D:\\图库根\\.gallery_index\\gallery",
     "added": 1234, "mode": "增量",
     "total_in_index": 98765, "secs": 30.1}
  ],
  "errors": []
}
```
`ok=false` 时配合 `errors[]` / `fatal_error` 查看原因；单根失败不影响其它根继续。
每步字段说明：
- `gallery_root`：本次实际使用的图库根目录（= 自动定位结果，或请求 root 自身）；
- `located`：是否经“祖先链自动定位”并入上层图库索引；
- `notices`：非致命提示（如：请求根自身有索引、而其上级另有图库宿主——疑似历史错位建库，见 4.5 节第 4 条）。

## 4.5 图库根自动定位（prefix 留空时）

索引的规范位置是 `<图库根>\.gallery_index\`。img_server 提交的 roots
可能是图库根自身，也可能是图库根**之下的某个子目录/子图集**（下载批
常落在子目录）。接收方处理规则：

1. 对每个请求 root，沿 **root 自身 → 逐级父目录**向上检查，
   取最近一个含本程序索引（`.gallery_index\*.meta.json`）的目录作为
   图库根宿主，增量只扫描请求 root 自身（不整根全扫），新内容并入该宿主的索引；
2. 整条祖先链都没有索引时，才把请求 root 自身当作新图库根，
   首次自动从零构建（`<root>\.gallery_index\gallery`）；
3. 请求里显式给了 `prefix` → 尊重请求，不做任何定位；
4. 若请求 root 自身就有索引、而其**上级**还存在另一个图库索引
   （典型：此前把下载子目录误当图库根建过索引），接收方仍并入最近的
   自身宿主，同时在 `notices` 给出提示——把子目录里的 `.gallery_index`
   删掉后再触发一次，就会自动并入上级图库索引。

示例（真实场景）：图库根 `F:\视频`（索引在 `F:\视频\.gallery_index\gallery`），
下载批落在 `F:\视频\Fenriruu-riru\...`。若请求 `roots=[F:\视频\Fenriruu-riru]`，
接收方会定位到 `F:\视频` 并增量并入其索引，而**不会**在
`Fenriruu-riru\.gallery_index` 下另建一套。

## 5. 手动/调试入口（不经按钮也能用）

```bat
:: CLI 无窗口执行（适合测试或自动化）：
python main.py ingest "D:\…\handoff\request_<id>.json"

:: 带界面自动执行（等价 gui 模式，可直接试）：
python gui.py --auto-handoff "D:\…\handoff\request_<id>.json"
:: 或先转 working 再交给 GUI（launcher gui 模式内部即此路径）
```
幂等：同一批 roots 重复触发多次，只有真正新增/变化的内容会入库。

## 6. 安全与不变量
- 接收侧**只读图片文件**；写入仅限：索引目录（图库内 `.gallery_index/`）与 handoff 目录；
- 索引一致性护栏照常生效：换解码器/参数后旧索引会提示重建，不会静默混用；
- launcher 超时不强制杀进程（尊重主程序处理流程），会写 `ok=false` 的 result 说明；
- 文件流转全程原子改名，双实例重入由 `request_→working_` 防住。

## 7. 版本
| 版本 | 变更 |
| --- | --- |
| v1（当前） | 初始协议：request/working/result 三态 + launcher + GUI auto 模式 + CLI ingest |
| v1.1 | 接收端新增**图库根自动定位**：roots 为子目录时自动并入上级图库索引（不再在子目录另建）；result 增加 `gallery_root/located/notices` 字段。img_server 侧无需改动 |
