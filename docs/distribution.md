# 图库分发插件（默认关闭）

## 用途

临时全量图库放在 NAS C（NVMe）上建索引；建完后把图片迁到你指定的多个目标盘/目录，
索引仍留在 C，只把图片移到目标位置。插件可随时撤掉：关闭 `distribute` 即恢复原行为。

## GUI

构建页：

- ☐ **构建后分发图库（默认关闭）** — 勾选才生效；
- **分发目标(路径=权重;多个用;分隔)**：
  ```
  Z:\gallery_a=1;Y:\gallery_b=1;D:\gallery_c=2
  ```
  权重越大分到越多文件；不填不会分发。
- 构建完成后自动迁移；进度/日志会显示。

## CLI

```powershell
python main.py build "C:\NAS-Share\ImageGalleryTemp" `
  --prefix "C:\NAS-Share\ImageGalleryTemp\.gallery_index\gallery" `
  --distribute `
  --distribute-target "D:\gallery_a=1" `
  --distribute-target "E:\gallery_b=1"
```

可选：

- `--distribute-source`：临时图库根（默认 `img_dir`）；
- `--distribute-verify-hash`：逐文件 SHA256（慢，但最稳）。

## 行为

1. 建库/写索引完成；
2. 读取索引路径数组，逐个把文件复制到目标，校验大小（可选 SHA256），成功后才删源；
3. 全部文件移动成功后才重写索引路径数组：
   - `*.paths.npy`
   - `*.fine_paths.npy`
   - `*_tiles.paths.npy` / `*_tiles.fine_paths.npy`（如果存在）
4. 每个文件写 `distribution_manifest_*.jsonl`；索引路径数组备份到
   `distribution_backup_*`；
5. 任一步失败：按 manifest 逆操作把已移动文件搬回，路径数组从备份恢复；
6. 索引前缀留在 C；图片分布在目标盘；查询沿索引路径直接找。

## 注意

- 目标目录必须与源目录不重叠；已存在的同名文件会拒绝覆盖；
- 分发期间不要手动改索引；
- 目标盘可用空间不足会在移动阶段失败并回滚；
- 索引分盘适配的“性能天平”后续会基于每个目标的实测吞吐/权重扩展，当前
  `path=weight` 是简单字节/权重天平。

## 回滚

分发 manifest 在索引目录：`distribution_manifest_*.jsonl`，路径数组备份在同目录
`distribution_backup_*`。回滚：

```powershell
python devtools/rollback_distribution.py --manifest "W:\ImageGalleryTemp\.gallery_index\distribution_manifest_....jsonl"
```

会按 manifest 逆序把文件搬回临时图库，并恢复 `*.paths.npy`/`*.fine_paths.npy`。
