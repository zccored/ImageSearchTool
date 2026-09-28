// ImageSearchTool · 图库检索管理器 — 前端入口
// Copyright (C) 2026 zccored
// 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
// This program is free software under the GNU Affero General Public License v3.0.

import { createApp } from 'vue';
import './theme.css';
import App from './App.vue';
import { boot } from './bridge.js';
import { initTheme } from './theme.js';
import {
  state, reduce, loadStrings, loadSchema, refreshStatus
} from './store.js';
import { installSelftest } from './selftest.js';

initTheme();                     // 恢复上次的亮/暗选择（index.html 里另有一份防闪的等价逻辑）
createApp(App).mount('#app');

// 先注册事件入口并通知 Python 就绪（早于此的事件不会丢，但也不会有事件），
// 再拉参数 schema / 索引状态（默认值一律来自 Python 侧的 Config）。
boot((ev) => reduce(ev)).then(async () => {
  state.ready = true;
  await loadStrings();
  await loadSchema();
  await refreshStatus();
  installSelftest();
});
