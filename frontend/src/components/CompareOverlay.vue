<script setup>
// ImageSearchTool · 大图对比（P2）：页内 overlay + 双区独立缩放/平移 + 拖放 + 全屏
// Copyright (C) 2026 zccored · 本程序是自由软件：AGPL-3.0-only（见仓库根 LICENSE）
//
// 与 tkinter 版 `compare_view.py` 对齐的行为：
//   * 两区独立缩放（滚轮以光标为中心）/ 平移（拖动）/ 双击复位；
//   * 底部成员条：点击把该图放到“活动区”，复选框与去重页同一份勾选状态；
//   * 拖放：把成员条的小图拖到某一区；把某一区的图拖到另一区 = 换到另一侧；
//   * `F11` 无边框全屏（Fullscreen API），`Esc` 关闭。
// 原图通过只读 `/image/<key>?p=…` 提供（服务端按“图库内/索引内”校验来源）。

import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import {
  state, t, humanBytes, closeCompare, setSide, moveToOther, zoomAt, panBy,
  fitSide, toggleSel, cmd
} from '../store.js';

const root = ref(null);
const urls = ref({ left: '', right: '' });
const cache = new Map();
const drag = ref(null);

async function urlFor(path) {
  if (!path) return '';
  if (cache.has(path)) return cache.get(path);
  const r = await cmd.imageUrl(path);
  cache.set(path, (r && r.url) || '');
  return cache.get(path);
}

async function refresh() {
  const [l, r] = await Promise.all([urlFor(state.compare.left), urlFor(state.compare.right)]);
  urls.value = { left: l, right: r };
}

watch(() => [state.compare.left, state.compare.right], refresh, { immediate: true });

function view(side) { return state.compare.view[side] || { s: 1, x: 0, y: 0 }; }
function styleFor(side) {
  const v = view(side);
  return { transform: `translate(${v.x}px, ${v.y}px) scale(${v.s})` };
}
function isSide(path, side) { return path && state.compare[side] === path; }
function memberOf(path) {
  return (state.compare.members || []).find((m) => m.path === path) || { thumb: '' };
}

function onWheel(side, e) {
  e.preventDefault();
  const box = e.currentTarget.getBoundingClientRect();
  const cx = e.clientX - box.left - box.width / 2;
  const cy = e.clientY - box.top - box.height / 2;
  zoomAt(side, e.deltaY < 0 ? 1.15 : 1 / 1.15, cx, cy);
}

function onDown(side, e) {
  state.compare.active = side;
  drag.value = { side, x: e.clientX, y: e.clientY };
}

function onMove(e) {
  if (!drag.value) return;
  panBy(drag.value.side, e.clientX - drag.value.x, e.clientY - drag.value.y);
  drag.value.x = e.clientX;
  drag.value.y = e.clientY;
}

function onUp() { drag.value = null; }

function onDblClick(side) { fitSide(side); }

function onStripDrag(e, path) {
  e.dataTransfer.setData('text/ise-path', path);
  e.dataTransfer.effectAllowed = 'copy';
}

function onPaneDrop(side, e) {
  e.preventDefault();
  const from = e.dataTransfer.getData('text/ise-move');
  if (from) {                       // 区→区：把该区的图放到另一侧
    if (from !== side) moveToOther(from);
    return;
  }
  const path = e.dataTransfer.getData('text/ise-path');
  if (path) setSide(side, path);
}

function onPaneDragStart(side, e) {
  e.dataTransfer.setData('text/ise-move', side);
  e.dataTransfer.effectAllowed = 'move';
}

async function toggleFullscreen() {
  try {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
    else await document.exitFullscreen();
  } catch (e) { /* 用户拒绝 / 不支持：忽略 */ }
  state.compare.fullscreen = !!document.fullscreenElement;
}

function onKey(e) {
  if (!state.compare.open) return;
  if (e.key === 'Escape') closeCompare();
  else if (e.key === 'F11') { e.preventDefault(); toggleFullscreen(); }
}

function onFsChange() { state.compare.fullscreen = !!document.fullscreenElement; }

onMounted(() => {
  window.addEventListener('keydown', onKey);
  document.addEventListener('fullscreenchange', onFsChange);
});
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey);
  document.removeEventListener('fullscreenchange', onFsChange);
});
</script>

<template>
  <div v-if="state.compare.open" class="overlay" ref="root"
       :class="{ fs: state.compare.fullscreen }">
    <div class="cmphead">
      <b>{{ t('cmp.title', '重复图大图对比') }}</b>
      <span class="chip">组 {{ state.compare.groupId }} · {{ state.compare.members.length }} 张</span>
      <span class="chip">{{ t('cmp.hint', '滚轮缩放 · 拖动平移 · 双击复位 · 拖小图到某区 / 把某区拖到另一区 · F11 全屏 · Esc 关闭') }}</span>
      <span class="spacer" />
      <button @click="fitSide(state.compare.active)">{{ t('cmp.fit', '复位(活动区)') }}</button>
      <button @click="toggleFullscreen">{{ state.compare.fullscreen ? t('cmp.exit_fs', '退出全屏') : t('cmp.fs', '全屏(F11)') }}</button>
      <button class="danger" @click="closeCompare">{{ t('cmp.close', '关闭(Esc)') }}</button>
    </div>

    <div class="cmpbody">
      <section v-for="side in ['left', 'right']" :key="side" class="pane"
               :class="{ active: state.compare.active === side }"
               @wheel="onWheel(side, $event)"
               @pointerdown="onDown(side, $event)"
               @pointermove="onMove"
               @pointerup="onUp" @pointerleave="onUp"
               @dblclick="onDblClick(side)"
               @dragover.prevent @drop="onPaneDrop(side, $event)">
        <header>
          <span class="pn">{{ side === 'left' ? '左' : '右' }}：{{ (state.compare[side] || '').split(/[\\/]/).pop() }}</span>
          <span class="chip">×{{ view(side).s.toFixed(2) }}</span>
          <span class="spacer" />
          <img class="panehandle" draggable="true" alt=""
               :src="memberOf(state.compare[side]).thumb"
               :title="t('cmp.drag', '拖到另一区 = 换到另一侧')"
               @dragstart="onPaneDragStart(side, $event)" />
        </header>
        <div class="paneimg">
          <img :src="urls[side]" alt="" :style="styleFor(side)" draggable="false" />
        </div>
      </section>
    </div>

    <div class="strip">
      <div v-for="m in state.compare.members" :key="m.path" class="stripitem"
           :class="{ active: isSide(m.path, 'left') || isSide(m.path, 'right') }"
           draggable="true" @dragstart="onStripDrag($event, m.path)"
           @click="setSide(state.compare.active, m.path)">
        <img :src="m.thumb" alt="" loading="lazy" />
        <div class="cap">
          <input type="checkbox" :checked="!!state.dedup.sel[m.path]" @click.stop="toggleSel(m.path)" />
          {{ m.name }}
        </div>
        <div class="cap2">{{ humanBytes(m.size) }}<template v-if="m.exact_copy"> · 字节相同</template></div>
      </div>
    </div>
  </div>
</template>
