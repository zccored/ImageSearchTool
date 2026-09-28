<script setup>
// ImageSearchTool · 图片列表页：扫描结果 + 已索引标记 + 预览 + 设为查询图
// Copyright (C) 2026 zccored · 本程序是自由软件：AGPL-3.0-only（见仓库根 LICENSE）

import { computed, onMounted, ref } from 'vue';
import { state, t, cmd, setLocation } from '../store.js';

const filter = ref('');
const selPath = ref('');
const previewUrl = ref('');
const previewInfo = ref('');

const rows = computed(() => {
  const kw = filter.value.trim().toLowerCase();
  const indexed = new Set(state.indexed);
  const out = [];
  for (const p of state.scan.paths) {
    if (kw && !p.toLowerCase().includes(kw)) continue;
    out.push({ path: p, indexed: indexed.has(norm(p)) });
  }
  return out;
});

function norm(p) {
  // 与 Python 侧 `os.path.normcase(os.path.abspath(...))` 对齐的口径（Windows）
  const s = p.replace(/\//g, '\\');
  return /^[a-z]:\\/i.test(s) ? s.toLowerCase() : s;
}

function basename(p) { return p.split(/[\\/]/).pop(); }

async function select(p) {
  selPath.value = p;
  const r = await cmd.thumbUrl(p);
  previewUrl.value = r.url || '';
  previewInfo.value = p;
}

function useAsQuery(p) {
  state.query = p;
  state.page = 'search';
}

async function openPath(p) { await cmd.openPath(p); }

onMounted(async () => { await setLocation(state.dir || null, state.prefix || null); });
</script>

<template>
  <div class="page list">
    <div class="row">
      <input class="grow" v-model="filter" :placeholder="t('list.filter', '按路径过滤…')" />
      <span class="chip">{{ t('list.count', '共') }} {{ rows.length }} / {{ state.scan.paths.length }}</span>
      <span class="chip">{{ t('list.indexed', '已入库') }} {{ state.indexed.length }}</span>
    </div>
    <div class="split">
      <div class="tablewrap">
        <table>
          <thead>
            <tr>
              <th class="c-idx">{{ t('list.col.indexed', '已索引') }}</th>
              <th>{{ t('list.col.name', '文件（双击=设为查询图）') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in rows" :key="r.path"
                :class="{ active: r.path === selPath }"
                @click="select(r.path)" @dblclick="useAsQuery(r.path)">
              <td class="c-idx">{{ r.indexed ? '✔' : '·' }}</td>
              <td :title="r.path">{{ basename(r.path) }}</td>
            </tr>
          </tbody>
        </table>
        <div v-if="!rows.length" class="empty">
          {{ t('list.empty', '还没有扫描结果：填好图库目录后点工具栏“扫描图库”') }}
        </div>
      </div>
      <aside class="preview">
        <img v-if="previewUrl" :src="previewUrl" alt="" />
        <div v-else class="ph">{{ t('list.preview', '（选中图片后在此预览）') }}</div>
        <div class="info">{{ previewInfo }}</div>
        <div class="row">
          <button :disabled="!selPath" @click="useAsQuery(selPath)">{{ t('list.use_query', '设为查询图') }}</button>
          <button :disabled="!selPath" @click="openPath(selPath)">{{ t('list.open', '打开原图') }}</button>
        </div>
      </aside>
    </div>
  </div>
</template>
