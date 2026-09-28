<script setup>
// ImageSearchTool · 以图搜图页：查询 + 三模式 + Top-K 网格（占位不丢格）+ 详情
// Copyright (C) 2026 zccored · 本程序是自由软件：AGPL-3.0-only（见仓库根 LICENSE）
//
// 网格规则（铁律）：`index === 命中下标`——缩略图解码失败也渲染占位，保证点哪张是哪张。

import { computed, ref, watch } from 'vue';
import { state, t, cmd } from '../store.js';

const props = defineProps({ perfSearch: { type: Boolean, default: false } });
const coarseK = ref(null);
const topK = ref(null);
const hybridWarned = ref(false);
const retryCount = ref({});      // 命中下标 -> 重试次数（缩略图 202 生成中时重取）

// 新一轮结果：清掉重试计数（键含路径，元素也会重建，避免上次的“失败态”残留）
watch(() => state.hits, () => { retryCount.value = {}; });

const detail = computed(() => {
  const i = state.selIndex;
  if (i < 0 || i >= state.hits.length) return null;
  const h = state.hits[i];
  const kindTxt = { tile: '局部命中(瓦片)', both: '整图+局部双命中', full: '整图命中' }[h.match_kind] || h.match_kind;
  const fine = Number.isFinite(h.fine_score) ? h.fine_score.toFixed(4) : 'n/a(未精排)';
  const box = h.box ? `    命中框: (${h.box[0]},${h.box[1]})-(${h.box[2]},${h.box[3]}) (原图像素)` : '';
  return { kindTxt, fine, box, h };
});

const summary = computed(() => {
  const r = state.result;
  if (!r) return t('search.no_result', '（尚无结果：选查询图后点“开始检索”；列表里双击任意图可直接设为查询）');
  const ms = (k) => `${((((r.times || {})[k]) || 0) * 1000).toFixed(1)}ms`;
  return `库 ${r.db} 张 | 候选 ${r.kept} 张 | 模式 ${r.method} | 合计 ${ms('total')}`
    + (r.self_excluded ? ' | 已剔除查询图自身' : '');
});

async function pick() {
  const r = await cmd.pickFile();
  if (r && r.path) state.query = r.path;
}

async function doSearch() {
  if (!state.query) return;
  if (state.mode === 'hybrid' && !hybridWarned.value) {
    hybridWarned.value = true;
    // 混合模式：两套索引并行打分，耗时约为单套 2 倍（与 tkinter 版同一提示口径）
    console.info('混合检索将同时载入整图与瓦片两套索引，耗时约为单套的 2 倍');
  }
  await cmd.search(state.query, state.mode, coarseK.value, topK.value, props.perfSearch);
}

function select(i) { state.selIndex = i; }

/** 缩略图未就绪（202 生成中）或失败：加时间戳重试（最多 6 次，累计约 5s）；仍失败则占位。 */
function retry(ev, i) {
  const n = (retryCount.value[i] || 0) + 1;
  retryCount.value = Object.assign({}, retryCount.value, { [i]: n });
  const img = ev.target;
  if (n <= 6) {
    const base = (state.hits[i].thumb || '').split('&r=')[0];
    setTimeout(() => { img.src = `${base}&r=${n}`; }, 250 * n);
  } else {
    img.style.visibility = 'hidden';
    img.parentElement.classList.add('failed');
  }
}

async function openHit() {
  if (detail.value) await cmd.openPath(detail.value.h.path);
}

async function copyHit() {
  if (!detail.value) return;
  const p = detail.value.h.path;
  try {
    await navigator.clipboard.writeText(p);
    state.statusText = `已复制路径：${p}`;
  } catch (e) {
    state.statusText = `复制失败：${e.message}`;
  }
}

async function exportSheet() {
  if (!state.hits.length) return;
  const r = await cmd.pickSave('search_result.png');
  if (!r || !r.path) return;
  const res = await cmd.exportSheet(state.hits.map((h) => h.path), r.path);
  state.statusText = res.ok ? `总览图已保存：${r.path}` : '总览图保存失败';
}
</script>

<template>
  <div class="page searchp">
    <div class="row">
      <label class="lab">{{ t('search.query', '查询图片') }}</label>
      <input class="grow" v-model="state.query" :placeholder="t('search.query_hint', '库内双击，或点“浏览…”选外部图片')" />
      <button @click="pick">{{ t('btn.browse', '浏览…') }}</button>
      <button class="primary" :disabled="state.busy || !state.query" @click="doSearch">
        {{ t('btn.search', '开始检索') }}
      </button>
    </div>
    <div class="row">
      <label class="lab">{{ t('search.mode', '检索模式') }}</label>
      <label class="chk"><input type="radio" value="full" v-model="state.mode" />整图索引 (Top≤10)</label>
      <label class="chk"><input type="radio" value="tiles" v-model="state.mode" />局部索引·瓦片</label>
      <label class="chk"><input type="radio" value="hybrid" v-model="state.mode" />混合·两者都搜 (Top≤20)</label>
      <label class="lab">{{ t('search.coarse_k', '粗筛候选') }}</label>
      <input class="num" type="number" min="10" v-model.number="coarseK" placeholder="默认" />
      <label class="lab">{{ t('search.top_k', '返回条数') }}</label>
      <input class="num" type="number" min="1" max="100" v-model.number="topK" placeholder="默认" />
    </div>
    <div class="row">
      <button :disabled="!state.hits.length" @click="openHit">{{ t('btn.open', '打开原图') }}</button>
      <button :disabled="!state.hits.length" @click="copyHit">{{ t('btn.copy', '复制路径') }}</button>
      <button :disabled="!state.hits.length" @click="exportSheet">{{ t('btn.export', '导出总览图…') }}</button>
      <span class="chip grow">{{ summary }}</span>
    </div>

    <div class="grid">
      <div v-for="(h, i) in state.hits" :key="i + '|' + h.path"
           class="cell" :class="{ active: i === state.selIndex }" @click="select(i)">
        <img :src="h.thumb" alt="" loading="lazy" @error="retry($event, i)" />
        <div class="ph">{{ t('search.no_preview', '无预览\n(超大图或解码失败)') }}</div>
        <div class="cap">
          {{ h.rank }}. {{ (h.match_kind === 'tile' ? '局部·' : (h.match_kind === 'both' ? '双·' : '')) }}{{ h.path.split(/[\\/]/).pop() }}
          <span class="cos">{{ Number.isFinite(h.fine_score) ? `cos=${h.fine_score.toFixed(3)}` : '(仅粗筛)' }}</span>
        </div>
      </div>
      <div v-if="!state.hits.length" class="empty">{{ summary }}</div>
    </div>

    <div class="detail">
      <template v-if="detail">
        <div>#{{ detail.h.rank }}  {{ detail.h.path }}</div>
        <div>类型={{ detail.kindTxt }}    ResNet相似度={{ detail.fine }}
          粗筛综合分={{ detail.h.coarse_score?.toFixed(4) }}
          指纹汉明比例={{ detail.h.d_fp?.toFixed(4) }}{{ detail.box }}</div>
      </template>
      <template v-else>{{ t('search.detail_hint', '点击结果缩略图查看详情（局部命中缩略图带红框）') }}</template>
    </div>
  </div>
</template>
