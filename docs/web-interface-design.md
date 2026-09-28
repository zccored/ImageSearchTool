# Web 版界面与服务层的设计说明（2026-09-27 一轮）

> **本文来源（如实声明）**：本文由**可核实的材料**重建 ——
> 作者那一轮的更新报告 `branch/CHANGES-2026-09-27.md`、仓库内的实际代码
> （`hybrid_search/service.py` / `gui_web.py` / `frontend/` / `image-search.spec`）、
> 以及接手方在 2026-09-28 合并后重跑的回归结果。
> **作者本机另留了一份不入库的设计过程记录**（含被否候选、逐条实测与阶段计划），
> 本文**不包含**那部分内容；若要补全，请把那份记录并进来（或告诉我路径，我来合）。
> 凡是本文没写依据的推断，都标了「（待补）」。
>
> 相关文档：`branch/CHANGES-2026-09-27.md`（原始报告）、`docs/HANDOFF_PROTOCOL.md`（交接协议）、
> `docs/perf-plan.md`（性能结论的唯一事实来源）、`frontend/README.md`（界面侧维护）、
> `packaging_README_分发说明.txt`（面向最终用户）。

---

## 一、目标与硬约束

| # | 目标 / 约束 | 落实方式 |
| :--- | :--- | :--- |
| 1 | 提供一个**更现代的界面**，但不改变检索/建库语义 | 界面只是壳；编排逻辑抽到 `hybrid_search/service.py`，与算法层解耦 |
| 2 | **两套界面并存**：tkinter 版保留为「保底」，不进垃圾桶 | `gui.py` 保留、净 −352 行，只留 UI 与线程调度 |
| 3 | **CLI 仍是主入口**，行为不变 | 子命令与参数名一个没增没删；只修了 2 个「默认值来源」缺陷（见 §五） |
| 4 | **不装 Node 也能改文案/配色** | 外置 `ui_strings.json` / `ui_theme.css`（放包根即可覆盖，缺失回落内置）；只有改布局才需要 `pnpm build` |
| 5 | 新界面**单独一个 exe** | `image-search.spec` 三入口：`ImageSearchGUI` / `ImageSearchWeb` / `ImageSearchCLI` |
| 6 | 新依赖**可整块摘除**，不影响 CLI | `requirements.txt` 里独立成组（`pywebview` / `pythonnet` / `bottle`）；删掉该组 + `frontend/` 即回到纯 CLI 形态 |
| 7 | 界面不得成为攻击面 | HTTP 只读、**只监听 `127.0.0.1`**；命令**不走 HTTP**，走进程内 `js_api`，且每个方法先校验 token |

## 二、分层：为什么要有 `hybrid_search/service.py`

抽取前，编排逻辑（扫描 / 建库 / 检索 / 去重 / 引擎缓存 / 性能画像 / 参数 schema）
写在 `gui.py` 里。要再接一套界面，只有两条路：**复制一份**（两份逻辑必然漂移）或**抽出来**（选它）。

服务层的契约（写在该文件 docstring 里）：

- **命令**：扫描 / 建库（整图·瓦片·compact）/ 检索（full·tiles·hybrid）/ 去重 / 引擎缓存与释放 / 性能画像 / 参数 schema；
- **每个命令带 `task_id`**：长任务可以并发、可以取消，事件能正确归属；
- **事件流**：`log` / `progress` / `phase_boundary(save|done)` / `viz_frame` / `perf_report` / `task_done` / `task_error`。

由此得到的关键性质：**换界面不影响索引**。tkinter 版与 Web 版调用的是同一份
`HybridEngine`，因此「同参数、同样本」建出来的索引**逐位一致**（实测见 §六）。

## 三、Web 版的实现形态

```
gui_web.py（Python 侧只有这一层壳）
├── 自建只读 WSGI（bottle）      仅暴露：前端 dist、/thumb/<key>、/image/<key>、外置文案主题、/health
├── Api（pywebview js_api）      命令入口；每个方法先校验 token；命令不经过 HTTP
├── EventBridge                  服务层事件 → 前端；含 numpy/dataclass 的 JSON 化兜底
└── ThumbService                 /thumb/<key> 复用 hybrid_search/thumbs.py 的 96px 磁盘缓存

frontend/（Vue3 + Vite，本质是一个本地网页）
└── 五页 + 页内大图对比覆盖层：图片列表 / 以图搜图 / 去重审查 / 参数 / 日志·可视化
```

几个刻意的选择：

- **不用 HTTP 传命令**：HTTP 段纯只读，写操作全部走进程内 `js_api`，绕开 CORS/端口/CSRF 这一类问题；
- **只监听回环**：回归里有一条 Gate 专门断言这一点（实测端口 `127.0.0.1:16368`）；
- **缩略图复用既有磁盘缓存**：不新造一套，`/thumb/<key>` 命中直接发；未命中时"就地生成 + 有上限等待"，超时返回 202 让前端重试（不阻塞 UI 线程）；
- **事件批量 JSON 化**：服务层返回的对象里有 numpy 数组与 dataclass（如命中框），必须兜住，否则整批事件会被丢掉。

### 免 Node 的维护路径

| 想改什么 | 怎么做（文件放**包根目录**） |
| :--- | :--- |
| 文案 / 窗口标题 | `ui_strings.json`：只写要覆盖的键（其余回落内置）；删掉 = 全用内置 |
| 配色 / 圆角 / 字号 | `ui_theme.css`：只写要覆盖的 CSS 变量（变量名见 `frontend/src/theme.css`）；**亮/暗两套分别写**（`:root` / `:root[data-theme="light"]`） |
| 布局 / 交互 | 改 `frontend/src/**` → `pnpm build` → 重新发布（**只有这一档需要 Node**） |

## 四、打包

- 三入口各自 `Analysis`，**共享** `binaries/datas`（避免依赖重复占盘）；
- `frontend/dist` 与外置文案随包；发布包里 `dist` 是**构建好的**，所以打包版不需要 Node；
- 未使用的 GUI 后端进 `excludes`（否则本机装着的 PySide6 会被整包收进去）；
- `SRC` 由 `ISE_SRC` / `SPECPATH` 推导，不硬编码绝对路径。

**包体随 torch 变体变化极大**：本机 **CPU-only torch** 打包 **778 MB**；
**CUDA 版 torch**（本机 `2.4.0+cu124`，2026-09-28 实测）为 **4,149 MB / 4,308 文件**。
**Web 层增量与显卡无关、基本恒定 ≈ 4.6 MB**（pythonnet + clr_loader + webview + `frontend/dist`）。

## 五、行为变化与兼容性（**这部分需要读者注意**）

### 5.1 CLI 的两个默认值来源被订正

抽取前的缺陷：`cli.py` 里两个 argparse 默认值会**覆盖** `Config` 默认值 ——

| 参数 | 改前 | 改后 | 后果 |
| :--- | :--- | :--- | :--- |
| `--png-decoder` | 默认 `"cv2"` 被当成显式传参 → meta 记 `cv2` | 默认 `None` = **不覆盖**（用 `Config` 的 `libdeflate`） | meta 口径与 GUI/Web 一致 → `stats` 等**终于能打开 CLI 自己建的索引** |
| `--fast-load` | `store_true`，不传也写回 `False` → CLI 建库写 **npz** | 只有显式传参才覆盖 | CLI 建库默认写**侧车 `.npy`**（可 mmap），与 GUI/Web 一致 |

修法是"**参数默认值只在 `config.py` 写一次**"，并把这条固化成回归：
`python -E devtools/verify_cli_defaults.py`（判据 `漂移: 0/11`）。

### 5.2 索引兼容矩阵

| 索引由谁建 | meta `png_decoder` | 存储 | 新 `stats`/`compact`/`bench` | 新 `search`/`add` | GUI / Web |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **旧 CLI**（≤ `1d97c07`） | `cv2` | npz | 见下注 | ✅ 加 `--png-decoder cv2` 即可 | ✗（GUI 也按 `libdeflate` 基准，属既有行为） |
| **新 CLI / GUI / Web** | `libdeflate` | sidecar | ✅ 不用加开关 | ✅ 不用加开关 | ✅ |

> **2026-09-28 补充（接手方实测）**：
> * `bench` **本来就已注册** feature 参数；
> * `compact` 走 `store.compact(prefix)`，**完全不读 `Config`**、也不做参数一致性校验，
>   因此它**不需要** feature 参数（实测 `compact` 对 cv2-meta 索引直接 rc=0）；
> * `stats` **确实缺**，已补上 `_add_feature_args()`。实测：
>   `stats --prefix <cv2-meta 索引>` → rc=2（报错正文明确指出 `png_decoder: 索引=cv2 当前=libdeflate`）；
>   `stats --prefix <同上> --png-decoder cv2` → **rc=0**。

### 5.3 日志不再"吞掉结尾"

PNG 噪音过滤（fd 2 接管成管道 + 泵线程）会**吞掉进程退出前的最后几行**——报错时终端只剩 INFO 行、
正文丢失。现改为日志直写真终端 fd，退出前再 `drain_stderr_noise()`。噪音过滤能力未变
（回归：51 行噪音被吞 / 1 行真实告警透出）。

## 六、验收与回归（判据与实测）

| # | 检查 | 判据 | 结果 |
| :--- | :--- | :--- | :--- |
| 1 | `python -E -m compileall -q .` | 退出码 0 | ✅ |
| 2 | `devtools/verify_cli_defaults.py` | `漂移: 0/11` + rc=0 | ✅（合并后复跑仍 0/11） |
| 3 | `devtools/verify_service.py --db 120` | 「结果: 全部通过」 | ✅（合并后复跑通过，10.8 s） |
| 4 | `devtools/verify_web_gui.py --no-window` | 「结果: 全部通过」 | ✅（9.0 s；只读路由 / 命令面 / 事件桥 / token 校验 / 去重删改后索引同步） |
| 5 | `devtools/verify_web_gui.py`（真窗口） | Gate 1/3/4/5 全过（含**只监听回环**） | ✅（14.6 s；含 G5 性能闸门：一屏 60 张缩略图 冷 219 ms / 热 92 ms） |
| 6 | `devtools/verify_phase_events.py` | 进度以 `save`→`done` 收尾 | ✅ |
| 7 | `devtools/verify_stderr_filter.py` | 51 行噪音被吞 / 1 行真实告警透出 | ✅ |
| 8 | 索引前后对拍 | 整图 `paths/hu/fp/md5s` 逐位一致；瓦片同键指纹逐位一致 | ✅ |
| 9 | 真包验收 | 三 exe 齐全；Web 只监听回环且 `/health`·`/` 全 200；GUI 起窗 8 s 存活；冻结 CLI `stats` rc=0 | ✅（2026-09-28 复现：`ImageSearchWeb.exe` 监听 `127.0.0.1:16368`，`/health` 200 / `/` 200） |

## 七、遗留与待决策

1. **`compact` 不需要 feature 参数**（已用代码与实测确认，见 §5.2 注）；若将来把一致性校验加进
   `store.compact`，这条结论要跟着重估。
2. **旧 `cv2`-meta 索引**的处置（用户手上有无这类索引）—— 三条路：查询/增量加 `--png-decoder cv2`；
   或 `build --force` 重建（推荐，顺便升级成侧车）；或统一 meta 口径后继续用。
3. `gui_web.py` 末尾曾有 20 行死代码（`if __name__ == "__main__":` 之后还跟着一个 `def _enrich`
   与 `# @@TAIL@@` 标记），是服务层抽取时留下的重复副本（同功能模块级函数 `enrich_search_result`
   已在正常工作）—— **2026-09-28 已删除**（858 → 838 行），Web 回归复跑仍全绿。
4. 作者本机那份**不入库的设计过程记录**（被否候选 / 逐条实测 / 阶段计划）尚未并入本文 —— 见文首声明。

## 八、复现命令

```bat
:: 服务层与界面回归
python -E devtools/verify_service.py --db 120
python -E devtools/verify_web_gui.py --no-window
python -E devtools/verify_web_gui.py                 :: 真窗口（含 Gate 1/3/4/5）
python -E devtools/verify_cli_defaults.py            :: 判据 漂移: 0/11

:: 从源码跑 Web 版（首次需构建前端；只改文案/配色不需要）
cd frontend && pnpm install && pnpm build && cd ..
python -E gui_web.py

:: 打包（三入口；frontend/dist 必须已构建）
python -E -m PyInstaller --noconfirm --clean image-search.spec ^
    --distpath <dist> --workpath <work>
```
