<script setup>
// ImageSearchTool · 去重审查页（P2）：分组 + 虚拟滚动 + 批量删除/移动 + 索引同步
// Copyright (C) 2026 zccored · 本程序是自由软件：AGPL-3.0-only（见仓库根 LICENSE）
//
// 与 tkinter 版（`gui.py` 的 DedupWindow）对齐的语义：
//   * ☑ = 将被处理（删除/移动），☐ = 保留；点成员行切换勾选；
//   * 勾选助手：保留最佳（每组第一张）/ 仅完全重复 / 未入库副本 / 全不选 / 反选；
//   * 删除默认走回收站（可还原）；移动保留相对图库根的目录结构；
//   * 勾选状态与对比窗双向联动（见 CompareOverlay）。
// 千组级数据的渲染方式：扁平行（组头 + 成员）+ 按固定行高的**虚拟滚动**，
// 只渲染可视区 ±5 行，缩略图仍由 `/thumb/<key>` 惰性加载。

import { computed, onMounted, ref } from 'vue';
import {
  state, t, humanBytes, dedupScan, dedupApply, selectKeepBest, selectExact,
  selectExactUnindexed, selectNone, invertSelection, toggleSel, selectedPaths,
  selectedBytes, openCompare, cmd
} from '../store.js';

const ROW = 42;
const box = ref(null);
const scrollTop = ref(0);
const viewH = ref(600);
const busy = computed(() => state.dedup.busy || state.busy);
const threshold = ref(state.dedup.threshold);

const rows = computed(() => {
  const rep = state.dedup.report;
  const out = [];
  if (!rep) return out;
  const kw = (state.dedup.filter || '').trim().toLowerCase();
  for (const g of rep.groups) {
    const members = g.members || [];
    if (state.dedup.onlySelected && !members.some((m) => state.dedup.sel[m.path])) continue;
    const shown = kw ? members.filter((m) => (m.name || '').toLowerCase().includes(kw)) : members;
    if (!shown.length) continue;
    out.push({ kind: 'group', key: `g${g.gid}`, g });
    shown.forEach((m, mi) => out.push({ kind: 'member', key: `m${g.gid}:${m.path}`, g, m, mi }));
  }
  return out;
});

const start = computed(() => Math.max(0, Math.floor(scrollTop.value / ROW) - 5));
const end = computed(() => Math.min(rows.value.length,
  Math.ceil((scrollTop.value + viewH.value) / ROW) + 5));
const visible = computed(() => rows.value.slice(start.value, end.value));
const offsetY = computed(() => start.value * ROW);

function onScroll(e) {
  scrollTop.value = e.target.scrollTop;
  const max = Math.max(0, rows.value.length * ROW - viewH.value);
  if (scrollTop.value > max) scrollTop.value = max;
}

const selected = computed(() => selectedPaths());
const selBytes = computed(() => selectedBytes());

async function runScan() {
  if (!state.scan.paths.length) {
    state.statusText = '请先在“图片列表”页扫描图库';
    return;
  }
  await dedupScan(state.scan.paths, threshold.value, state.prefix);
}

async function doDelete() {
  const paths = selected.value;
  if (!paths.length) return;
  const ok = await confirmDialog('删除到回收站',
    `将把 ${paths.length} 张图移入 Windows 回收站（可还原），合计 ${humanBytes(selBytes.value)}。\n`
    + `索引同步：${state.dedup.sync ? '开' : '关'}。\n\n继续？`);
  if (!ok) return;
  await dedupApply('delete', paths);
}

async function doMove() {
  const paths = selected.value;
  if (!paths.length) return;
  const dest = await pickDir('选择目标图库位置（保留相对目录结构）');
  if (!dest) return;
  await dedupApply('move', paths, { dest, baseRoot: state.dir });
}

/** 确认框：优先用 pywebview 原生对话框；自动测试可注入 `window.__ise_confirm`。 */
async function confirmDialog(title, message) {
  if (typeof window.__ise_confirm === 'function') return !!window.__ise_confirm(title, message);
  try {
    const r = await cmd.confirm(title, message);
    return !!(r && r.ok);
  } catch (e) { return false; }
}

async function pickDir(title) {
  if (typeof window.__ise_pick_dir === 'function') return window.__ise_pick_dir(title);
  const r = await cmd.pickDir();
  return (r && r.path) || '';
}

function groupIndex(g) { return state.dedup.report ? state.dedup.report.groups.indexOf(g) : -1; }

function onRowDblClick(r) {
  const gi = groupIndex(r.g);
  if (gi >= 0) openCompare(gi, r.m ? r.m.path : '');
}

function toggleGroup(g) {
  const members = g.members || [];
  const on = members.some((m) => state.dedup.sel[m.path]);
  members.forEach((m) => { state.dedup.sel[m.path] = !on; });
  state.dedup.sel = Object.assign({}, state.dedup.sel);
}

function isIndexed(m) {
  const p = String(m.path || '').toLowerCase();
  return state.indexed.includes(p);
}

onMounted(() => { if (box.value) viewH.value = box.value.clientHeight || 600; });
</script>

<template>
  <div class="page dedup"
       :data-members="state.dedup.report ? state.dedup.report.n_images : 0"
       :data-groups="state.dedup.report ? state.dedup.report.groups.length : 0"
       :data-sel="selected.length">
    <div class="row">
      <label class="lab">{{ t('dedup.threshold', '近似阈值(%)') }}</label>
      <input class="num" type="number" min="0.2" max="20" step="0.2" v-model.number="threshold" />
      <label class="chk"><input type="checkbox" v-model="state.dedup.sync" />{{ t('dedup.sync', '同步索引(prune)') }}</label>
      <button class="primary" :disabled="busy" @click="runScan">{{ t('dedup.scan', '开始查验重复') }}</button>
      <span class="chip">{{ t('dedup.sel', '已勾选') }} {{ selected.length }} 张 · {{ humanBytes(selBytes) }}</span>
      <span class="chip" v-if="state.dedup.report">
        {{ state.dedup.report.groups.length }} 组 / {{ state.dedup.report.n_images }} 张 ·
        完全 {{ state.dedup.report.n_exact }} / 近似 {{ state.dedup.report.n_near }} ·
        可释放 {{ humanBytes(state.dedup.report.wasted_bytes) }}
      </span>
    </div>
    <div class="row">
      <button data-act="keep-best" :disabled="!state.dedup.report" @click="selectKeepBest">{{ t('dedup.keep_best', '保留最佳(每组首张)') }}</button>
      <button data-act="exact" :disabled="!state.dedup.report" @click="selectExact">{{ t('dedup.only_exact', '仅完全重复') }}</button>
      <button data-act="exact-unindexed" :disabled="!state.dedup.report" @click="selectExactUnindexed">{{ t('dedup.exact_unindexed', '未入库的完全副本') }}</button>
      <button data-act="none" :disabled="!state.dedup.report" @click="selectNone">{{ t('dedup.none', '全不选') }}</button>
      <button data-act="invert" :disabled="!state.dedup.report" @click="invertSelection">{{ t('dedup.invert', '反选') }}</button>
      <button data-act="delete" class="danger" :disabled="busy || !selected.length" @click="doDelete">{{ t('dedup.delete', '删除到回收站') }}</button>
      <button data-act="move" :disabled="busy || !selected.length" @click="doMove">{{ t('dedup.move', '移动到…') }}</button>
      <input class="grow" v-model="state.dedup.filter" :placeholder="t('dedup.filter', '按文件名过滤…')" />
      <label class="chk"><input type="checkbox" v-model="state.dedup.onlySelected" />{{ t('dedup.only_sel', '只看含勾选') }}</label>
    </div>

    <div class="vlist" ref="box" @scroll="onScroll">
      <div class="vspacer" :style="{ height: (rows.length * ROW) + 'px' }">
        <div class="vwin" :style="{ transform: `translateY(${offsetY}px)` }">
          <div v-for="r in visible" :key="r.key" class="vrow"
               :class="r.kind" :style="{ height: ROW + 'px' }"
               @dblclick="onRowDblClick(r)">
            <template v-if="r.kind === 'group'">
              <button class="mini" @click.stop="toggleGroup(r.g)">⇄</button>
              <b class="gtitle">{{ r.g.title }}</b>
              <span class="chip">{{ r.g.all_exact ? '完全重复' : '近似重复' }}
                · 汉明 {{ (r.g.max_hamming * 100).toFixed(2) }}%</span>
              <span class="chip">可释放 {{ humanBytes(r.g.wasted_bytes) }}</span>
              <span class="chip" v-if="r.g.min_cos != null">最低余弦 {{ Number(r.g.min_cos).toFixed(4) }}</span>
            </template>
            <template v-else>
              <input type="checkbox" :checked="!!state.dedup.sel[r.m.path]"
                     @click.stop="toggleSel(r.m.path)" />
              <img class="t96" :src="r.m.thumb" alt="" loading="lazy" />
              <span class="name" :title="r.m.path">{{ r.m.name }}</span>
              <span class="chip">{{ humanBytes(r.m.size) }}</span>
              <span class="chip" v-if="r.m.exact_copy">字节相同</span>
              <span class="chip">汉明 {{ (r.m.hamming * 100).toFixed(2) }}%
                <template v-if="r.m.cos != null">· cos {{ Number(r.m.cos).toFixed(3) }}</template>
              </span>
              <span class="chip" :class="{ on: isIndexed(r.m) }">
                {{ isIndexed(r.m) ? '✔ 已入库' : '· 未入库' }}
              </span>
              <span class="spacer" />
              <button class="mini" @click.stop="cmd.openPath(r.m.path)">{{ t('btn.open', '打开原图') }}</button>
              <button class="mini" @click.stop="openCompare(groupIndex(r.g), r.m.path)">
                {{ t('dedup.compare', '对比…') }}
              </button>
            </template>
          </div>
        </div>
      </div>
      <div v-if="!rows.length" class="empty">
        {{ state.dedup.report ? t('dedup.empty', '没有符合筛选条件的重复组')
           : t('dedup.hint', '点“开始查验重复”：完全重复按 MD5，近似重复按二值指纹汉明（复用索引里的特征，不重新解码）') }}
      </div>
    </div>
  </div>
</template>
