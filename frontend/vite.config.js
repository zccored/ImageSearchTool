// ImageSearchTool · 图库检索管理器 — 前端构建配置（Vue3 + Vite）
// Copyright (C) 2026 zccored
//
// 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
// 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
// This program is free software under the GNU Affero General Public License
// v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 构建约束：
//   outDir 必须避开仓库根的 dist/（PyInstaller 也用 dist/）→ 固定输出 frontend/dist
//   base 用相对路径：产物同时适用于自建 WSGI（`/`）与任何子路径托管
export default defineConfig({
  plugins: [vue()],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
    sourcemap: false
  },
  server: { port: 5273, strictPort: false }
})
