# docs/perf —— 性能图（本地生成，不随仓库分发）

本目录原先存放全量建库/基准的性能报告截图与报告本体（PNG / HTML / JSON）。
公开发布版已移除这些产物，原因：报告由本机真实图库生成，页面内含本机绝对路径与
文件名等隐私信息。

要自己生成同类性能图：

```bat
:: 1) 建库时会自动在本机 perf_reports/ 下生成 gui_index_*.html|json（GUI 里点建库即可）
:: 2) 基准对照（可选，需自备 devtools/ 下的基准脚本）
python -E perfscope.py            :: 本仓库自带的性能作用域工具
```

判读口径、完整实测数据与被证伪的假设见 `docs/perf-plan.md`。