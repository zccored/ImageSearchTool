# 贡献指南（约定 / 验证 / 提交前检查）

> 本项目 AI 与人类协作者共用同一套约定。本文只写**公开可查**的规则 —— 原来散落在本机
> 开发笔记里的约定，凡是对改代码有约束力的，都已收录到这里。
>
> 目录导航见 `README.md`「七、目录结构」；性能口径见 `docs/perf-plan.md`；
> 交接协议见 `docs/HANDOFF_PROTOCOL.md`。

---

## 0. 三条铁律（违反必出错）

1. **唯一实现**：检索 / 建库 / 索引读写**只有 `hybrid_search/` 一份**；界面（`gui.py`、
   `gui_web.py` + `frontend/`）只做 UI 与线程调度，一律调 `hybrid_search/service.py`
   （前端无关的唯一编排层），**禁止把编排或算法逻辑复制进界面**。
2. **默认值只写一次**：全部可调参数的默认值**只在 `hybrid_search/config.py` 的 `Config`**
   （图片扩展名在 `DEFAULT_EXTENSIONS`）。新增参数必须 **Config 字段 + CLI 开关 + GUI 控件**
   三处同步；参数页一律从服务层的 `get_config_schema()` 读，**前端不得复制默认值**。
   ⚠️ **CLI 的 argparse 默认值必须等于 `Config` 默认、或写成"不覆盖"**
   （`default=None` / `store_true` + 仅显式传参才覆盖）—— 否则会出现"写进索引 meta 的参数
   ≠ 读索引时的基准"，导致**自己建的索引打不开**。
3. **图库图片只读**：任何功能只准写索引目录 / 缓存，**绝不动用户原图**；
   `pyinstall_runtime_*.py` 与冻结态相关的代码不得依赖 `stdout/stderr`（窗口模式它们是 `None`）。

## 1. 目录职责（一句话版）

| 路径 | 职责 | 不许做 |
| :--- | :--- | :--- |
| `hybrid_search/config.py` | 所有参数与默认值 | 别处写默认值字面量 |
| `hybrid_search/engine.py` | 两级检索流水线 + 索引生命周期 | 别在界面里重写流水线 |
| `hybrid_search/service.py` | **前端无关编排层**（命令带 `task_id`，事件流） | 别在这里做 UI / 线程创建 |
| `hybrid_search/{coarse,fine,tile_index,store,dedup,prep_cache,thumbs,visuals}.py` | 算法与存储 | 别把算法下沉到界面 |
| `hybrid_search/cli.py` | argparse 子命令 | 别让 argparse 默认值改 Config |
| `gui.py` | tkinter 界面（**保底**，主线是 Web 版） | 别加编排逻辑 |
| `gui_web.py` + `frontend/` | Web 界面（主线） | 同上；前端只经 `bridge.js` 调 api |
| `devtools/` | 开发期回归 / 基准脚本 | 别把一次性调试脚本留在里面 |

## 2. 代码规范

- **源文件头**：全部 `.py`、`image-search.spec`、`frontend/**` 的源码与配置都带
  **中文 + 英文双语 AGPL-3.0-only 声明**（照抄 `hybrid_search/config.py` 头部）。
  本项目**不使用** SPDX `SPDX-License-Identifier:` 行。保留各文件既有的 BOM 与否、换行用 CRLF。
- **torch / torchvision 必须在函数内延迟导入**（PyInstaller 静态分析看不到，靠 spec 的
  `hiddenimports` 收集）；不要提到模块顶层。
- **日志**：业务代码统一用 `hybrid_search/io_utils.py` 的 `LOGGER`；**不要 `print`**
  （冻结态窗口模式没有 stdout）。CLI 进度走 `progress.CliPhaseProgress`。
- **性能开关**：一律做成 `Config` 字段，注释里附「实测数字 + 开关名 + 出处」；
  新结论与**被证伪的假设**都回写 `docs/perf-plan.md`（它是性能口径的唯一事实来源）。
- **索引位置**只由推导函数给出：整图 `<图库根>\.gallery_index\gallery`，瓦片一律
  `tile_index.tiles_prefix_of(prefix)` —— **不要手拼字符串**。

## 3. 提交前检查（按改动面选）

```bat
python -E -m compileall -q .                 :: 必过：语法
python -E devtools/verify_service.py         :: 服务层端到端（真实链路 + 事件契约）
python -E devtools/verify_web_gui.py --no-window   :: Web 界面回归（无头）
python -E devtools/verify_web_gui.py         :: Web 界面回归（含真窗口 Gate 1/3/4/5）
python -E devtools/verify_phase_events.py    :: 进度必须以 save/done 阶段边界收尾
python -E devtools/verify_stderr_filter.py   :: libpng 噪音过滤（吞噪音 / 透告警）
python -E devtools/verify_cli_defaults.py    :: 改了 CLI 参数默认值后必跑：判据「漂移: 0/11」
python -E make_test_dataset.py --out ./test_data --db 5000   :: 造模拟图库
```

按数据特性选断言（**别硬套逐位比较**）：

| 数据 | 断言方式 |
| :--- | :--- |
| 粗筛侧（`fp` / `hu` / `md5s` / `paths`） | 可逐位比对 |
| 瓦片索引 | 落盘顺序 = 线程完成顺序（每次不同）→ 按「原图路径 + 框」建键做**顺序无关**比对 |
| 精排特征（FP16） | 有 ~3e-7 余弦抖动（GPU autocast 非确定性）→ 用容差，别要求逐位相等 |

## 4. 界面相关（Web 版）

- 改**文案 / 配色**不需要 Node：把 `ui_strings.json` / `ui_theme.css` 放包根（或仓库根）即可覆盖，
  见 `README.md`「〇·八」与 `frontend/README.md`。
- 改**布局 / 交互**需要 Node 工具链：

```bat
cd frontend && pnpm install && pnpm build    :: 产物 frontend/dist（不入库，随包分发）
python -E gui_web.py                         :: 源码运行（需先构建前端）
```

- 两条硬性约束（都有过真实故障）：① 必须等 `pywebviewready` 再碰 `window.pywebview.api.*`；
  ② `js_api` 方法按**位置**传参（不能用 keyword-only 参数）。

## 5. 提交信息

- 一句话说清"改了什么 + 为什么"，多个关注点拆成多条提交；
- 涉及性能的提交请带上**口径**（哪台机器、是否 CPU-only、前后数字出自哪份报告）；
- 不要提交本地产物：索引目录、`prep_cache/`、`thumbs/`、`*.npz`、`perf_reports/`、
  `frontend/node_modules/`、`frontend/dist/`、`dist/`、`__pycache__/`（已在 `.gitignore`）。

## 6. 许可

AGPL-3.0-only（见 `LICENSE`）。**修改后若通过网络对外提供服务，同样视为分发**，
须提供对应版本的完整源码（第 13 条）。随包分发的第三方组件清单见
`frontend/LICENSE-NOTICE` 与 `requirements.txt` 的分组注释。
