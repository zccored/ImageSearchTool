═══════════════════════════════════════════════════════════════
  图库检索管理器 ImageSearch —— 独立工具包使用说明
  （二值法粗筛 + ResNet 精排的混合图库检索系统）
═══════════════════════════════════════════════════════════════

一、本工具包含什么
  ImageSearchGUI.exe   图形界面：扫描图库 → 建索引 → 以图搜图（tkinter 版）
  ImageSearchWeb.exe   新界面（Web 版，pywebview + Vue3，见第六节）
  ImageSearchCLI.exe   命令行版（无窗口脚本/自动化用，见第四节）
  _internal\           运行依赖、内置模型权重、前端产物（勿删、勿动）

  ※ 三个 exe 只在使用当前版打包配置（image-search.spec）构建的包里齐全；
    早期分发包（2026-09-07 及更早）只有 ImageSearchGUI.exe 与 ImageSearchCLI.exe。

二、系统要求
  * Windows 10/11 64 位，无需安装 Python 或任何依赖；
  * 内存建议 8GB+（5000 张图库约需 2~3GB）；
  * 显卡：**不限定品牌**。包内自带的是 CUDA 版加速库 ——
      有 NVIDIA 显卡 → 自动用 GPU；
      其他显卡（AMD / Intel 核显）或驱动不匹配的机器 → 自动退回 CPU，
      功能完全一致，只是建库/检索明显更慢；
      （程序按 `device=auto` 判定：能用 CUDA 就用，否则用 CPU。）
  * 无需联网（内置 ResNet18 预训练权重；选 ResNet50 等模型时
    需要联网自动下载对应权重）。

三、图形界面用法（推荐）
  1) 双击 ImageSearchGUI.exe；
  2) 顶部“图库目录”填/浏览你的图片文件夹；
  3) “扫描图库”→ 勾选需要的内容 → “① 全部入库并建索引”
     （或“③ 增量入库”只加新图）；
  4) 右下“选择查询图”（库内双击，或浏览外部图片）→“开始检索”，
     Top-K 缩略图网格直接显示，点选可查看详情/打开原图/复制路径。

四、命令行用法
  在 ImageSearch 目录打开终端（或把 ImageSearchCLI.exe 加入 PATH）：
    ImageSearchCLI.exe build <图库目录> --prefix <索引前缀>
    ImageSearchCLI.exe add   <新增图片目录> --prefix <索引前缀>   :: 增量去重
    ImageSearchCLI.exe build-tiles <图库目录> --prefix <瓦片索引前缀>  :: 瓦片(局部)索引
    ImageSearchCLI.exe add-tiles   <图库目录> --prefix <瓦片索引前缀>
    ImageSearchCLI.exe search <查询图> --prefix <索引前缀> --top-k 10
    ImageSearchCLI.exe search <局部截图> --mode tiles --tiles-prefix <瓦片索引前缀>
    ImageSearchCLI.exe stats --prefix <索引前缀>
    ImageSearchCLI.exe eval  <查询图目录> --prefix <索引前缀>     :: 召回率评估
    ImageSearchCLI.exe compact --prefix <索引前缀>                :: npz → 侧车 .npy
  示例：
    ImageSearchCLI.exe build D:\图片库 --prefix D:\图片库\.gallery_index\gallery
    ImageSearchCLI.exe search D:\某张图.jpg --prefix D:\图片库\.gallery_index\gallery

五、索引位置与格式
  默认索引前缀 <图库根>\.gallery_index\gallery（整图）与 …\gallery_tiles（瓦片局部），
  新索引默认写成"侧车 .npy"（paths/fp/hu/fine/… 并列 .npy，可 mmap 快载，打开快、
  常驻内存小），meta.json 记录建库参数；旧 npz 索引仍可读取，用 compact 可就地转换。
  相同参数下索引可直接跨机器复制使用（路径需一致）。
  图库文件本身只读，索引重建/增量均幂等安全。

六、新界面（Web 版，可选）
  ImageSearchWeb.exe   新界面（pywebview + Vue3）：图片列表 / 以图搜图 /
                       去重审查 / 参数 / 日志·可视化 + 大图对比
  * 与图形界面（第 1 节）功能等价，二选一即可；老的 ImageSearchGUI.exe 保留，
    新界面出问题随时切回（两者共用同一套检索实现与索引）。
  * 首次启动需要 **WebView2 运行时**（Windows 11 自带；Windows 10 若提示缺失，
    到 Microsoft 官网下载 “WebView2 Runtime (Evergreen)” 安装一次即可，离线可分发）。
  * 本工具会在 **本机回环地址（127.0.0.1）随机端口** 起一个**只读**静态服务器，
    仅用于把界面文件与缩略图发给内置浏览器；所有会改动数据的操作（建索引、
    删除/移动重复图等）都不经过网络，而是同进程直调。企业安全软件若记录
    “Python 监听回环端口”，属预期行为。
  * 改文案/配色不需要开发工具：程序目录下的
      ui_strings.json   界面文字（只写要覆盖的键）；窗口标题也在里面（app.title）
      ui_theme.css      配色/圆角/字号（覆盖 CSS 变量）；亮/暗两套要分别写：
                        暗色 :root { … }、亮色 :root[data-theme="light"] { … }
    改完重启新界面即生效；删掉这两个文件就回到内置默认。
  * 界面页签行右端可切换亮色 / 暗色（按钮显示“点它会切到哪个”），选择会记住。
  * 新界面的检索逻辑与 CLI/图形界面完全一致（同一 hybrid_search/service.py），
    “参数”页给出的等价 CLI 命令可直接复制到命令行执行。

七、其它
  * “切换启动：全栈图库管理器”按钮仅在本机存在
    D:\code\新的代码\全栈图库管理器 v3.2bata\main.py 时激活；
    作为独立工具包分发到其它机器时该按钮自动禁用（属预期）。
  * 首次运行如被杀毒软件拦截：此为 PyInstaller 单目录打包的正常
    误报，添加信任后运行即可（本工具不含任何网络外联）。
  * 数据安全：本工具只读取图片、只写索引文件，不修改你的图片。

八、开源协议
  本工具采用 GNU Affero 通用公共许可证 v3.0（AGPL-3.0-only），完整条款见随包
  LICENSE 文件。分发本工具（含 exe 与 _internal 目录）时请一并附上 LICENSE 全文
  与对应版本的完整源码；用它对外提供网络服务同样视为分发（AGPL 第 13 条）。
  新界面（Web 版）的第三方依赖（Vue 3 / Vite 等，均为 MIT 许可）随 _internal
  内的前端产物分发，许可说明见 frontend/LICENSE-NOTICE。
  源码地址：https://github.com/zccored/ImageSearchTool
