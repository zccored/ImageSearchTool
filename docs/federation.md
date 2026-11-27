# 图库联邦 / 链式索引插件（默认关闭）

## 概念

| 名称 | 位置 | 作用 |
|---|---|---|
| 总索引目录 | `federation_total_dir`（默认索引前缀目录） | `federation.json` + `federation_trace.json`；汇总所有节点、索引文件、断点信息 |
| 分发节点目录 | 每个分发目标根，如 `D:\kemono` | `.image-search-federation.json`（node marker）；记录本盘区域、索引文件哈希/行数、断点/分发信息、链到其它节点 |

节点 marker 之间通过 `linked_nodes` 互链；从任意一个节点出发可以递归发现整条链。
总索引目录的 `federation.json` 则给出全部节点的快照，方便一次校验。

## GUI

构建页新增：

- ☐ **启用图库联邦/链式索引（插件，默认关闭）**
- **总索引目录(联邦插件，可空)**
- **联邦节点 ID(可选)**

只有勾选才写 manifest；不勾选完全走原逻辑。

## CLI

```powershell
python main.py build "C:\NAS-Share\ImageGalleryTemp" `
  --prefix "C:\NAS-Share\ImageGalleryTemp\.gallery_index\gallery" `
  --distribute `
  --distribute-target "D:\kemono=1" `
  --federation `
  --federation-total "C:\NAS-Share\ImageGalleryTemp\.gallery_index" `
  --federation-node-id "disk-D-kemono"
```

## 读取行为

### 从分发盘读取

如果选择的目录有 `.image-search-federation.json`，`service.scan` 会自动：

1. 读取本节点 marker；
2. 沿 `linked_nodes` 递归发现同链节点；
3. 只保留当前存在的节点根目录；
4. 把这些根目录的图片合并扫描/建库。

因此选择 `D:\kemono` 时，会自动把 `E:\kemono2` 等链上目录拼成综合图库。

### 从总索引目录读取

`federation.json` 列出所有节点。`devtools/federation_tool.py --total ... --verify`
会逐个检查：

- 节点根目录是否存在；
- marker 是否存在；
- marker 中记录的索引文件是否存在；
- 节点断点/分发信息。

任何节点不可读，校验结果标记 `available=false` / `ok=false`，该区域不参与扫描；
检索结果中路径已不存在的命中也会被过滤（`federation_dropped`）。

## 索引文件追溯

每个 node marker 的 `index_files` 记录：

```json
{"path": "...gallery.paths.npy", "bytes": 123, "rows": 37782,
 "sha256": "...", "mtime": "..."}
```

`resume` 记录分发/断点信息（目标、源、已移动数量、manifest 路径、时间）。
总目录的 `federation_trace.json` 再汇总一遍，方便审计。

## 当前边界

本轮只做 **manifest / 发现 / 校验 / 过滤** 层：

- 特征文件仍是总索引那一份，没有为每个盘复制/切分特征；
- 总索引所在盘不可用时，节点 marker 只能证明“这里曾经有哪些文件”，
  还不能单独查询（后续可做 per-node shard index）；
- 插件关闭时，不写 marker、不扫描链、不过滤命中，核心路径不变。

## 工具

```powershell
python devtools/federation_tool.py --start "D:\kemono" --json
python devtools/federation_tool.py --total "C:\...\.gallery_index" --verify
```
