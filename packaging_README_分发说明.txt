═══════════════════════════════════════════════════════════════
  图库检索管理器 ImageSearch —— 独立工具包使用说明
  （二值法粗筛 + ResNet 精排的混合图库检索系统）
═══════════════════════════════════════════════════════════════

一、本工具包含什么
  ImageSearchGUI.exe   图形界面：扫描图库 → 建索引 → 以图搜图
  ImageSearchCLI.exe   命令行版（无窗口脚本/自动化用，见第四节）
  _internal\           运行依赖与内置模型权重（勿删、勿动）

二、系统要求
  * Windows 10/11 64 位，无需安装 Python 或任何依赖；
  * 内存建议 8GB+（5000 张图库约需 2~3GB）；
  * 显卡：包内含 CUDA 加速库——有 NVIDIA 显卡自动用 GPU；
    无 NVIDIA 显卡自动退回 CPU（速度较慢但可用）。
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
    ImageSearchCLI.exe search <查询图> --prefix <索引前缀> --top-k 10
    ImageSearchCLI.exe stats --prefix <索引前缀>
    ImageSearchCLI.exe eval  <查询图目录> --prefix <索引前缀>     :: 召回率评估
  示例：
    ImageSearchCLI.exe build D:\图片库 --prefix D:\图片库\.gallery_index\gallery
    ImageSearchCLI.exe search D:\某张图.jpg --prefix D:\图片库\.gallery_index\gallery

五、索引位置与格式
  默认索引前缀 <图库根>\.gallery_index\gallery(.meta.json/.coarse.npz/
  .fine.npz)。相同参数下索引可直接跨机器复制使用（路径需一致）。
  图库文件本身只读，索引重建/增量均幂等安全。

六、其它
  * “切换启动：全栈图库管理器”按钮仅在本机存在
    D:\code\新的代码\全栈图库管理器 v3.2bata\main.py 时激活；
    作为独立工具包分发到其它机器时该按钮自动禁用（属预期）。
  * 首次运行如被杀毒软件拦截：此为 PyInstaller 单目录打包的正常
    误报，添加信任后运行即可（本工具不含任何网络外联）。
  * 数据安全：本工具只读取图片、只写索引文件，不修改你的图片。
