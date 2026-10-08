# 新界面（Web 版）· 维护说明

> 面向**不写前端**的维护者：日常改文案 / 改配色**不需要 Node**；只有改布局/交互才需要。

## 1. 它是什么

- `gui_web.py`（Python 壳，pywebview）+ `frontend/`（Vue3 + Vite 产物）。
- 所有检索 / 建库 / 去重逻辑都在 **`hybrid_search/service.py`**（唯一编排层）；
  界面只做两件事：调 `window.pywebview.api.*`（命令）与渲染事件（日志 / 进度 / 可视化帧）。
- 旧的 `gui.py`（tkinter）**保留**：新界面出问题随时切回（两条路都走同一个服务层）。

### 为什么是 pywebview + Vue3（选型结论，别再重新评估）

| 候选 | 结论 |
| :--- | :--- |
| **pywebview 6 + Vue3** | ✅ 采用：复用系统 **WebView2**，随包只有 ~4.6 MB 增量；Python 侧零框架、命令走进程内 `js_api`（不经 HTTP） | 
| Qt WebEngine（PySide/PyQt） | ❌ 体积与许可成本都高（示例：装了 PySide6 的机器上，单 `Qt6WebEngineCore` 一项就 ~195 MB，还得额外 `excludes` 排除） |
| Electron / Tauri | ❌ 要随包带 Node 运行时或 Rust 工具链，与本项目"单目录独立工具包"的取向冲突 |
| 纯浏览器（本地起服务 + 手动开浏览器） | ❌ 要用户自己开浏览器、没有原生窗口与图标，体验断层 |
| 继续用 tkinter | ➖ 保留为**保底**：零额外依赖，但复杂交互（虚拟滚动、双区对比）写起来代价高 |

> 结论：**Web 版是图形界面主线，tkinter 版是保底**；两者共用 `service.py`，随时可切。

## 2. 三条不用 Node 的维护路径

| 想改什么 | 怎么做 | 需要 Node？ |
| :--- | :--- | :--- |
| **文案** | 改包根目录的 `ui_strings.json`（只写要覆盖的键，其余回落内置；删掉该文件=全用内置）—— **含系统窗口标题**（`app.title`，由 `gui_web.py` 读取；页内不重复显示产品名） | 否 |
| **配色 / 圆角 / 字号** | 在包根目录放 `ui_theme.css`，只写要覆盖的 CSS 变量（见 `frontend/src/theme.css` 里的 `:root`）。**亮/暗两套要分别写**：暗色 `:root { … }`、亮色 `:root[data-theme="light"] { … }`（后者更具体，不会被前者盖掉） | 否 |
| **主题切换按钮** | 界面自带（页签行右端「☀ 亮色 / 🌙 暗色」），选择记在浏览器本地存储里；不需要配置 | 否 |
| **后端行为**（检索 / 建库 / 参数） | 照旧改 Python（`hybrid_search/*`、`hybrid_search/service.py`）；改完刷新界面即可 | 否 |

> 两个外置文件都由 `gui_web.py` 的只读路由提供（`/ui_strings.json`、`/ui_theme.css`），
> 不需要重新构建；不存在时自动回落内置默认。

## 3. 改布局 / 交互（需要 Node）

```bash
cd frontend
pnpm install          # 首次
pnpm dev              # 浏览器里调样式（此时没有 Python 桥，api 调用会报错，属预期）
pnpm build            # 产物固定输出到 frontend/dist（不要改到仓库根的 dist/，那是 PyInstaller 的）
```

构建产物 `frontend/dist` **不入库**，发布打包时随包携带（`image-search.spec` 把它和 `ui_strings.json` /
`ui_theme.css` 一起打进 `_internal/`；给最终用户的说明见 `packaging_README_分发说明.txt` 第六节）。

## 4. 目录

```
frontend/
├─ index.html            入口（含外置主题 <link>）
├─ vite.config.js        outDir=frontend/dist；base='./'
└─ src/
   ├─ main.js            挂载 App；boot(事件入口) → 拉 schema → 拉状态
   ├─ bridge.js          pywebview 桥：ready()/call()/run()/onEvent()（**所有 api 调用的唯一出口**）
   ├─ store.js           reactive 状态 + 事件 reducer + 命令封装（cmd.* / dedupScan / dedupApply / openCompare …）
   ├─ selftest.js        自检钩子（`__ise_selftest` / `__ise_selftest_dedup` / `__ise_selftest_compare`，`devtools/verify_web_gui.py` 用）
   ├─ theme.css          内置主题（CSS 变量）
   ├─ App.vue            工具栏 / 页签（列表·检索·去重·参数·日志）/ 状态栏 / 对比覆盖层
   ├─ components/VizPanel.vue        处理过程可视化（canvas，24fps 节流在 Python 侧）
   ├─ components/CompareOverlay.vue  大图对比（双区缩放/平移/拖放/F11/Esc）
   └─ pages/{LibraryList,Search,Dedup,Params,Logs}.vue
```

### 页面对应的 Python 能力

| 页面 | 命令（`Api` / `cmd`） | 说明 |
| :--- | :--- | :--- |
| 图片列表 | `scan` / `indexed_paths` / `thumb_url` | ✔/· 已索引标记来自索引里的路径集合 |
| 以图搜图 | `search`（full/tiles/hybrid） | 网格 `index === 命中下标`；缩略图 202 时自动重试 |
| 去重审查 | `dedup_scan` / `dedup_delete` / `dedup_move` | 删除走回收站；`sync` 决定是否 `prune` 索引；报告就地更新 |
| 大图对比 | `image_url`（只读 `/image/<key>`）+ `thumb_url` | 原图经服务端来源校验；对比是页内 overlay（无新窗口） |
| 参数 | `get_config_schema` / `get_config` / `set_config` / `cli_command` | 默认值只来自 `hybrid_search/config.py` |
| 日志 | 事件流（`log`/`progress`/`phase_boundary`/`viz_frame`/`perf_report`） | 阶段边界必须以 `save`/`done` 收尾 |

## 5. 架构与通道分工（谁走 HTTP、谁走 js_api）

```
前端 dist ──WebView2──┬── 只读 HTTP（本地回环，仅 127.0.0.1）：
                      │     /                  前端产物（dist）
                      │     /thumb/<key>[?p=]  缩略图（96px 磁盘缓存）
                      │     /image/<key>[?p=]  原图（大图对比用）
                      │     /ui_strings.json   外置文案
                      │     /ui_theme.css      外置主题
                      │     /health            探活
                      └── js_api（进程内直调，不经 HTTP；Windows 走原生 WebMessageReceived）
                              Api ── 每个方法先校验 `webview.token`
                                  ▼
                        hybrid_search/service.py（唯一编排层）
```

- **一切变异与查询走 `Api`**（scan / build / search / dedup_* / get|set_config / …）；
  HTTP 段**只读**，且路径都要过「索引内路径集合」校验（`?p=` 只用于生成缩略图，
  路径永不回显给前端）。
- **事件**：`service` 事件 → 队列 → 后台线程批量 `window.evaluate_js("window.__ise_event([...])")`；
  可视化帧按「最近一帧优先」节流（等价 tkinter 版的有界队列 + 丢中间帧）。
  事件名与载荷的唯一契约在 `hybrid_search/service.py` 的 docstring。
- **窗口**只监听回环（`devtools/verify_web_gui.py` 的 Gate 3 会验证）；冻结态不依赖 stdout。

## 6. Gate 与回归

`python -E devtools/verify_web_gui.py [--no-window] [--db N] [--keep]`

| 判据 | 内容 |
| :--- | :--- |
| 无头（默认） | 只读路由、命令面、事件契约、错误 token 被拒、去重删除/移动后索引同步 |
| **Gate 1** | 真窗口起窗 + `js_api` 往返 |
| **Gate 3** | HTTP **只监听 127.0.0.1** |
| **Gate 4** | 错误 token 的调用被拒 |
| **Gate 5** | 一屏缩略图全部落定（吞吐 / 失败计数，前端自检钩子 `__ise_selftest`） |
| **Gate 6** | 亮/暗主题两套配色都真的生效（`body` 背景色不同）+ 点击按钮即切（真窗口；前端自检钩子 `__ise_selftest_theme`） |

改动界面后至少跑一次无头；涉及桥/窗口/静态资源时跑真窗口模式。

## 7. 打包

- `image-search.spec` 有**三个入口**：`ImageSearchGUI` / `ImageSearchWeb` / `ImageSearchCLI`；
  Web 入口把 `frontend/dist` 打进 `_internal/frontend/dist`，并把包根的
  `ui_strings.json` / `ui_theme.css`（存在才带）放进 `_internal/`。
- **打包前必须先 `pnpm build`**；`frontend/dist` **不入库**。
- 运行期要求系统 **WebView2** 运行时（Win10 1803+ 一般已内置）。
- 给最终用户的说明见 `packaging_README_分发说明.txt` 第六节。

## 8. 亮/暗色主题

- 页签行右端有切换按钮（「☀ 亮色」/「🌙 暗色」= 点它会切到哪个）；选择存 `localStorage['ise.theme']`，
  下次启动沿用（`index.html` 里有一份防闪的内联恢复逻辑）。
- 实现只有三处：`src/theme.js`（切 `<html data-theme>`）、`src/theme.css`（两套变量）、
  `App.vue` 的一个按钮；**组件样式一律走 CSS 变量**，所以配色只有变量一个出口。
- 覆盖配色时**两套要分别写**（外置 `ui_theme.css`）：

```css
/* 暗色（默认主题） */
:root { --bg: #101418; --accent: #58a6ff; }
/* 亮色 */
:root[data-theme="light"] { --bg: #fafbfc; --accent: #1f6fb2; }
```

- 回归：`devtools/verify_web_gui.py`（真窗口）的 **Gate 6** 会断言两套配色**确实生效**
  （`body` 背景色不同）以及**点击按钮即切**；只跑无头模式时这条会跳过。

## 9. 两条硬性注意事项

1. **必须先等 `pywebviewready`** 再碰 `window.pywebview.api.*`（`bridge.ready()` 已封装；
   任何组件都**不要**在 `setup` 里直接调 `window.pywebview.api`）。
2. **不能在界面里复制参数默认值**：参数页一律读 `get_config_schema()`（默认值只在
   `hybrid_search/config.py` 写一次）。

## 10. 许可

`frontend/**` 的源文件带与项目一致的**中英双语 AGPL-3.0-only 声明**（见各文件头与
`frontend/LICENSE-NOTICE`）。
