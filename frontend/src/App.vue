<script setup>
// ImageSearchTool · 图库搜索管理器 — 主壳：工具栏 / 页签 / 状态栏
// Copyright (C) 2026 zccored · 本程序是自由软件：AGPL-3.0-only（见仓库根 LICENSE）
//
// 界面只做两件事：调 js_api（store.cmd）与渲染状态（store.state）；
// 检索/建库/去重逻辑全部在 hybrid_search/service.py —— 界面只调 js_api 与渲染状态，
// 不得把编排逻辑复制到这里。

import { computed, ref } from 'vue';
import { state, t, setLocation, cmd } from './store.js';
import { currentTheme, toggleTheme } from './theme.js';
import LibraryList from './pages/LibraryList.vue';
import Search from './pages/Search.vue';
import Params from './pages/Params.vue';
import Logs from './pages/Logs.vue';
import Dedup from './pages/Dedup.vue';
import CompareOverlay from './components/CompareOverlay.vue';

const pages = {
  list: LibraryList, search: Search, dedup: Dedup, params: Params, logs: Logs
};
const tabs = computed(() => ([
  { key: 'list', label: t('tab.list', '图片列表') },
  { key: 'search', label: t('tab.search', '以图搜图') },
  { key: 'dedup', label: t('tab.dedup', '去重审查') },
  { key: 'params', label: t('tab.params', '参数') },
  { key: 'logs', label: t('tab.logs', '日志 / 可视化') }
]));
const perfBuild = ref(false);
const perfSearch = ref(false);
const verify = ref(false);

// 亮/暗色：状态只存 <html data-theme>（配色在 theme.css），这里只负责显示与切换
const theme = ref(currentTheme());
const themeLabel = computed(() => (theme.value === 'dark'
  ? t('btn.theme_light', '☀ 亮色')
  : t('btn.theme_dark', '🌙 暗色')));
function onToggleTheme() {
  theme.value = toggleTheme();
}

async function pickDir() {
  const r = await cmd.pickDir();
  if (!r || !r.path) return;
  state.dir = r.path;
  await setLocation(state.dir, null);
}

function can(action) {
  if (state.busy) return false;
  if (action === 'scan' || action === 'build' || action === 'tiles') return !!state.dir;
  if (action === 'add') return !!state.dir && state.scan.count > 0;
  if (action === 'compact') return state.status.full_exists || state.status.tiles_exists;
  return true;
}

async function runScanner() {
  await setLocation(state.dir, state.prefix || null);
  await cmd.scan(verify.value);
}
async function runBuild() {
  await cmd.build(state.scan.paths, true, perfBuild.value);
}
async function runAdd() {
  await cmd.add(state.scan.paths, perfBuild.value);
}
async function runTiles() {
  await cmd.tiles(state.scan.paths, state.status.tiles_exists, perfBuild.value);
}
async function runCompact() {
  await cmd.compact(perfBuild.value);
}
async function release() {
  const r = await cmd.release();
  state.statusText = `已释放索引内存（RSS 回落 ${Math.round(r.freed_mb || 0)} MB）`;
}

async function openPerf() {
  const r = await cmd.latestPerf();
  if (r && r.path) await cmd.openPath(r.path);
  else state.statusText = '还没有生成过性能图：在“参数”页勾选导出开关后跑一次任务';
}
</script>

<template>
  <div class="app">
    <section class="toolbar">
      <label class="lab">{{ t('toolbar.dir', '图库目录') }}</label>
      <input class="dir" v-model="state.dir" :placeholder="t('toolbar.dir_hint', '选择或粘贴图库根目录')" />
      <button @click="pickDir">{{ t('btn.browse', '浏览…') }}</button>
      <label class="chk"><input type="checkbox" v-model="verify" />{{ t('toolbar.verify', '校验可解码(慢)') }}</label>
      <button :disabled="!can('scan')" @click="runScanner">{{ t('btn.scan', '扫描图库') }}</button>
      <button :disabled="!can('build')" @click="runBuild">{{ t('btn.build', '① 全部入库并建索引') }}</button>
      <button :disabled="!can('add')" @click="runAdd">{{ t('btn.add', '② 增量入库') }}</button>
      <button :disabled="!can('tiles')" @click="runTiles">{{ t('btn.tiles', '③ 子图(瓦片)索引') }}</button>
      <button :disabled="!can('compact')" @click="runCompact">{{ t('btn.compact', '优化索引存储') }}</button>
      <button :disabled="state.busy" @click="release">{{ t('btn.release', '释放索引内存') }}</button>
      <button @click="openPerf">{{ t('btn.perf', '最近性能图') }}</button>
    </section>

    <section class="perfflags">
      <label class="chk">
        <input type="checkbox" v-model="perfBuild" />
        {{ t('toolbar.perf_build', '索引阶段导出性能图(有性能损耗)') }}
      </label>
      <label class="chk">
        <input type="checkbox" v-model="perfSearch" />
        {{ t('toolbar.perf_search', '搜图阶段导出性能图(有性能损耗)') }}
      </label>
      <span class="chip">{{ t('toolbar.index', '索引') }}: {{ state.status.prefix || '—' }}</span>
      <span class="chip">{{ t('toolbar.storage', '存储') }}: {{ state.status.storage || '—' }}
        ({{ state.status.n || 0 }} / {{ state.status.tiles_n || 0 }})
      </span>
      <span class="chip">{{ t('toolbar.scan_info', '扫描') }}: {{ state.scan.count }} 张</span>
      <span class="chip">{{ t('toolbar.indexed', '已入库') }}: {{ state.indexed.length }} 张</span>
    </section>

    <nav class="tabs">
      <button v-for="tb in tabs" :key="tb.key"
              :class="{ active: state.page === tb.key }"
              :disabled="state.busy && tb.key !== 'logs'"
              @click="state.page = tb.key">
        {{ tb.label }}
      </button>
      <span class="spacer" />
      <button class="theme" :title="t('btn.theme_hint', '切换亮色 / 暗色（记住选择；配色可用 ui_theme.css 覆盖）')"
              @click="onToggleTheme">{{ themeLabel }}</button>
    </nav>

    <main class="body">
      <component :is="pages[state.page]" :perf-search="perfSearch" />
      <CompareOverlay />
    </main>

    <footer class="status">
      <span class="badge" :class="state.busy ? 'busy' : 'idle'">
        {{ state.busy ? t('app.busy', '处理中…') : t('app.idle', '空闲') }}
      </span>
      <span class="text">{{ state.statusText }}</span>
      <span class="ready">{{ state.ready ? t('app.ready', '通道已连接') : t('app.connecting', '连接中…') }}</span>
    </footer>
  </div>
</template>
