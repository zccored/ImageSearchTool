// ImageSearchTool · 图库检索管理器 — 构建后处理：把随产物分发的第三方许可文本拷进 dist
// Copyright (C) 2026 zccored
// 本程序是自由软件：AGPL-3.0-only（见仓库根 LICENSE）。
//
// 为什么需要：前端产物是编译后的 JS，Vue 的 MIT 许可文本不会自动出现在里面；
// 对外分发工具包时（AGPL 义务 + 第三方许可）应当让许可文本随产物一起走。

import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const dist = join(root, 'dist');

// 只拷“会被编译进产物”的运行时依赖（Vite 是构建期依赖，不进包）
const targets = [
  ['vue', 'vue/LICENSE', 'Vue-LICENSE.txt'],
];

mkdirSync(dist, { recursive: true });
let copied = 0;
for (const [pkg, rel, out] of targets) {
  const src = join(root, 'node_modules', pkg, rel.slice(pkg.length + 1));
  if (!existsSync(src)) {
    console.warn(`[copy-licenses] 跳过（未找到 ${pkg} 的许可文件）: ${src}`);
    continue;
  }
  copyFileSync(src, join(dist, out));
  copied += 1;
  console.log(`[copy-licenses] ${out} <- node_modules/${pkg}`);
}
console.log(`[copy-licenses] 完成，拷贝 ${copied} 个许可文件到 dist/`);
