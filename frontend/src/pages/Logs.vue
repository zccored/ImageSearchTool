<script setup>
// ImageSearchTool · 日志 / 可视化页：实时日志 + 阶段进度 + 24fps 处理过程可视化
// Copyright (C) 2026 zccored · 本程序是自由软件：AGPL-3.0-only（见仓库根 LICENSE）
//
// 数据来源全是事件（log / progress / phase_boundary / viz_frame / perf_report），
// 与 tkinter 版共用同一事件契约（字段表见 hybrid_search/service.py 的 docstring）。

import { computed, nextTick, ref, watch } from 'vue';
import { state, t, cmd } from '../store.js';
import VizPanel from '../components/VizPanel.vue';

const box = ref(null);
const onlyWarn = ref(false);

const rows = computed(() => (onlyWarn.value
  ? state.logs.filter((l) => /WARN|ERROR/.test(l.level))
  : state.logs));

const tasks = computed(() => Object.entries(state.progress).map(([id, p]) => ({
  id,
  done: p.done, total: p.total, phase: p.phase, boundary: p.boundary,
  pct: p.total ? Math.round((p.done / p.total) * 100) : 0
})));

const phases = computed(() => state.schema ? (state.schema.pages && null) : null);

watch(() => state.logs.length, async () => {
  await nextTick();
  if (box.value) box.value.scrollTop = box.value.scrollHeight;
});

async function openReport(p) { await cmd.openPath(p); }
</script>

<template>
  <div class="page logs">
    <div class="row">
      <span class="chip">{{ t('logs.count', '日志') }} {{ state.logs.length }}</span>
      <label class="chk"><input type="checkbox" v-model="onlyWarn" />{{ t('logs.only_warn', '只看告警/错误') }}</label>
      <span class="chip" v-for="tk in tasks" :key="tk.id">
        {{ tk.phase }} {{ tk.done }}/{{ tk.total }} ({{ tk.pct }}%)
        <b v-if="tk.boundary">· {{ tk.boundary }}</b>
      </span>
    </div>
    <div class="split">
      <div class="logbox" ref="box">
        <div v-for="(l, i) in rows" :key="i" class="line" :class="l.level">
          <span class="t">{{ l.time }}</span>{{ l.text }}
        </div>
      </div>
      <aside class="viz">
        <div class="vtitle">{{ t('logs.viz', '处理过程可视化 · 24fps') }}</div>
        <VizPanel :frame="state.viz" />
        <div class="vmeta">
          {{ state.viz.kind === 'coarse' ? '① 粗筛·二值点阵'
             : (state.viz.kind === 'fine' ? '② ResNet·采样象限' : '等待首帧…') }}
          · 帧 #{{ state.viz.frame }}
        </div>
        <div class="perf">
          <div class="vtitle">{{ t('logs.perf', '性能图（参数页勾选开关后生成）') }}</div>
          <div v-for="(p, i) in state.perf.slice(-6)" :key="i" class="perfrow">
            <a href="#" @click.prevent="openReport(p.path)">{{ p.stage }} · {{ p.path.split(/[\\/]/).pop() }}</a>
            <span v-if="p.modal" class="tag">{{ t('logs.modal', '已提示') }}</span>
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>
