可以尝试阅读D:\code\新的代码\全栈图库管理器 v3.2bata\image-search中的项目，但不要进行任何更改，下面是另一个智能体的对话以及必要说明情况：

对话完成情况:
备份:D:\code\备份\image-search-handoff-v1-20260907-000106(交接落地后快照,排除 __pycache__、perf_reports),最终 compileall rc=0,pycache 已清理。

接收端设计(协议流程)
img_server 侧按钮只需做三件事:写请求 JSON → 以分离进程方式启动 handoff_launcher.py
 → 立即安全退出自己。之后的完整流程由我方代码自动接管:

复制
img_server 按钮(只负责写 + 启动 + 退出)
   │  ① 原子写 handoff/request_<id>.json
   │  ② Popen("pythonw", "handoff_launcher.py", "<该json路径>", creationflags=DETACHED) → 立刻退出
   ▼
handoff_launcher.py(独立常驻进程,等待期不阻塞 img_server)
   │  ③ psutil 轮询等待 expect_exit(pids/names)全部退出 → 确认 img_server 已让出资源
   │  ④ 记录 working_<id>.json
   ▼
image-search GUI 以 --auto-handoff 启动
   │  ⑤ App 后台线程逐 root 处理:首个请求首次建索引 → 增量 add(幂等,路径+MD5 去重)
   │  ⑥ 任一 root 成功则全量合并保存;完成后原子写 result_<id>.json(ok/error/统计)
   │  ⑦ 进度经既有消息泵走(fused/coarse/fine 各阶段,GUI 侧两个 24fps 可视化队列照常工作)
新增/改动文件(位于 image-search\)
文件	作用
hybrid_search/handoff.py	协议核心:schema v1 校验、working_ 幂等标记、run_ingest(首次遇到自动建索引)、原子结果回写
handoff_launcher.py
进程等待(psutil,支持 pid/进程名)→ CLI 直跑或 mark_working + Popen 拉起 GUI 自动模式
gui.py
App(..., auto_handoff=) + 后台 worker + 结果泵,静默加载不弹"请先扫描图库"
hybrid_search/cli.py	新增 main.py ingest <request.json> 子命令(供 CLI 分支)
docs/HANDOFF_PROTOCOL.md	完整协议文档(时序、JSON 各字段、幂等/安全语义)
docs/img_server_接入需求.md	给 img_server 智能体的对接要求(可整份复制)
README.md
新增"〇·六 跨进程自动交接"章节
端到端验证证据(临时库,均已清理)
CLI 分支:首次建 60/60 张 ≈3.2s;重复触发 added 0(幂等);
GUI --auto-handoff 分支:产出 result_request_ht3.json,"ok": true;
普通无参启动回归正常;
失败分支写入 "ok": false + error 信息(请求文件与索引目录留现场,方便排查)。
请复制粘贴给 img_server 智能体的材料
文件路径:D:\code\新的代码\全栈图库管理器 v3.2bata\image-search\docs\img_server_接入需求.md

这份文档已写明对方所需的一切:按钮放置位置建议、请求 JSON 的完整字段(roots、gallery_name、prefix、png_decoder、expect_exit 的 pid/names)、原子写文件的代码骨架(含 .tmp + os.replace)、分离启动 handoff_launcher.py
 的命令(进程名禁止写成 python.exe,以免误等)、以及"img_server 必须先退出、GUI 后启动"的顺序约束。

我方代码已全部就绪,对方只需按该文档在 img_server 上加按钮。对接后如需我用真实 img_server 进程做一次联合演练,随时可以安排。



必要说明情况：
# 给 img_server.py 智能体的接入需求（复制粘贴稿）

你好，请为 img_server 侧实现“下载完成 → 交接给图库检索管理器”的按钮与回调。
接收端（image-search）已按下方协议实现并通过端到端测试，你只需在 img_server
一侧完成三件事：**① 写交接文件 ② 启动过渡进程 ③ 安全退出**。协议全文见
image-search/docs/HANDOFF_PROTOCOL.md；下面是直接可用的最小实现。

---

## 0. 常量

```python
import hashlib, json, os, subprocess, sys, time, threading

HANDOFF_DIR = r"D:\code\新的代码\全栈图库管理器 v3.2bata\handoff"
LAUNCHER = (r"D:\code\新的代码\全栈图库管理器 v3.2bata\"
            r"image-search\handoff_launcher.py")
```

## 1. 按钮回调（在 img_server 的 UI 上新增“下载完成，交给图库检索”按钮）

```python
def on_handoff_button_clicked(self, download_roots, extra_pids=()):
    """
    download_roots : list[str]  本次批次下载/校验完成的内容根目录（可多个）
    extra_pids     : 本进程之外需要等待退出的 pid（一般填上级主程序 pid）
    调用时机：UI 按钮按下。此时 img_server 的下载与 hash 校验均已全部完成。
    """
    os.makedirs(HANDOFF_DIR, exist_ok=True)
    request_id = "batch_" + time.strftime("%Y%m%d_%H%M%S")
    req = {
        "schema": 1,
        "kind": "download_batch_complete",
        "request_id": request_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "img_server",
        "roots": [{"path": p, "note": "batch"} for p in download_roots],
        "prefix": "",                      # 留空=接收端自动定位图库根（见文末“重要更新”）
        "open_mode": "gui",                # 默认打开图库检索管理器并自动增量建库
        "expect_exit": {
            "pids": [os.getpid()] + list(extra_pids),
            "names": [],                   # 也可填具体 exe 名；禁止 python.exe
        },
        "note": "",
    }
    fp = os.path.join(HANDOFF_DIR, f"request_{request_id}.json")
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(req, f, ensure_ascii=False, indent=2)
    os.replace(tmp, fp)                    # 原子落盘

    # 先启动过渡进程（detached：img_server 退出不影响它继续等待）
    DETACHED = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, "-X", "utf8", LAUNCHER, fp, "--wait", "90"],
        cwd=os.path.dirname(LAUNCHER),
        creationflags=DETACHED,
        close_fds=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 然后安全退出 img_server 与上级主程序（务必在“启动过渡进程之后”）
    # —— 释放 80%+ 内存，为图库检索管理器让路：
    # 请按你方现有主程序结构调用退出（关窗口/任务队列/保存配置/退出循环），
    # 例如 self.close(); QApplication.quit() 或 os._exit(0)。
```

要点：
- **先写文件 → 再 Popen 过渡进程 → 最后退出**（顺序不可颠倒）；
- request 幂等：重复批次、重复触发都安全（接收方按内容去重）；
- 不要等待过渡进程（它 detached 会自行等待你的进程退出）；
- `names` 若填写请用真实 exe 名（如你方打包后的 exe）；跑源码调试时留空 names、
  只填 pids，或不要勾选“等待退出”的强约束（把过渡进程 --wait 当超时保护即可）。

## 2. 验收清单（对方自测）

1. 图库目录放 20+ 张新图 → 点按钮 → img_server 写 `request_*.json`、窗口退出；
2. 过渡进程日志应出现“等待 img_server 退出…”→ 随后图库检索管理器窗口打开；
3. GUI 日志区出现“自动交接：校验并增量建库”，进度条走完后状态栏显示新增张数；
4. `handoff\result_<id>.json` 存在且 `ok: true`，steps[0].added == 新图张数；
5. 再次点按钮（图未变）→ `added: 0`（幂等验证）；
6. 取消勾选/异常时：过渡进程超时写 `ok:false`，不强制结束任何进程。

## 3. 你可能需要的辅助信息

- 图库检索管理器正常启动命令：`python gui.py`（不受影响）；
- 手动无界面增量：`python main.py ingest <request.json>`;
- 索引自动放在 `<图库根>\.gallery_index\gallery.*`，首次接收会自动从零构建，
  后续为增量；索引参数默认值（PNG=cv2、GPU 自动、粗筛 64 网格等）。
- 有任何字段疑问/版本升级请保持 `schema=1` 字段向后兼容。

## 4. 重要更新：接收端“图库根自动定位”（你方无需改代码，但请知悉）

**下载根可以是图库根，也可以是图库根下的任意子目录**——接收端会沿
目录的祖先链自动向上查找含 `.gallery_index\*.meta.json` 的最近目录，
把增量并入**该图库根的既有索引**，不再（错误地）在子目录里另建一套索引。
例：图库根 `F:\视频`、下载落在 `F:\视频\Fenriruu-riru`，即便请求
`roots` 只填后者，增量也会并入 `F:\视频\.gallery_index\gallery`。

因此你方按钮回调里无需计算“图库根在哪”，把本次批次实际落盘的目录
（可深可浅）原样填进 `roots` 即可；`prefix` 恒留空。
result JSON 中每步带 `gallery_root`（实际并入的图库根）与
`located`（是否自动定位）字段，验收清单第 4 条可直接核对
`steps[0].gallery_root` 是否符合预期。
