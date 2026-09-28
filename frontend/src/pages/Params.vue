<script setup>
// ImageSearchTool · 参数页：完全由服务层 schema 驱动（默认值只在 Config 写一次）
// Copyright (C) 2026 zccored · 本程序是自由软件：AGPL-3.0-only（见仓库根 LICENSE）
//
// 交互约定（T1：参数页与 tkinter 版同源）：
//   * 字段名/默认值/范围/提示全部来自 `get_config_schema()`（不在前端复制默认值）；
//   * 保存走 `set_config()`（服务层按 schema 校验，越界会被拒绝并回显原因）；
//   * 同时给出“等价 CLI 命令”提示，方便过渡到命令行。

import { computed, onMounted, ref, watch } from 'vue';
import { state, t, applyConfig, loadSchema, refreshCli } from '../store.js';

const draft = ref({});
const msg = ref('');
const cliOp = ref('build');

const pages = computed(() => (state.schema && state.schema.pages) || []);
watch(() => state.config, (c) => { draft.value = Object.assign({}, c); }, { immediate: true });
watch(cliOp, () => refreshCli(cliOp.value));

function setVal(key, v) { draft.value = Object.assign({}, draft.value, { [key]: v }); }

async function save() {
  const changed = {};
  for (const [k, v] of Object.entries(draft.value)) {
    if (state.config[k] !== v) changed[k] = v;
  }
  if (!Object.keys(changed).length) { msg.value = t('params.nochange', '没有改动'); return; }
  try {
    const r = await applyConfig(changed);
    msg.value = `${t('params.saved', '已应用')}: ${JSON.stringify(r.applied)}`;
    await refreshCli(cliOp.value);
  } catch (e) {
    msg.value = `${t('params.reject', '参数被拒绝')}: ${e.message}`;
  }
}

async function reload() {
  await loadSchema();
  draft.value = Object.assign({}, state.config);
  msg.value = t('params.reloaded', '已从服务层重新读取默认值');
}

onMounted(async () => { if (!state.schema) await loadSchema(); await refreshCli(cliOp.value); });
</script>

<template>
  <div class="page params">
    <div class="row">
      <button class="primary" @click="save">{{ t('params.save', '应用参数') }}</button>
      <button @click="reload">{{ t('params.reload', '恢复为服务层默认') }}</button>
      <span class="chip">{{ msg }}</span>
    </div>
    <div class="cols">
      <div v-for="p in pages" :key="p.key" class="col">
        <h3>{{ p.title }}</h3>
        <div v-for="g in p.groups" :key="g.title" class="group">
          <div class="gtitle">{{ g.title }}</div>
          <div v-for="f in g.fields" :key="f.key" class="field" :title="f.tip">
            <label class="flab">{{ f.label }}<span v-if="f.clr_adv" class="adv">·高级</span></label>
            <template v-if="f.kind === 'bool'">
              <input type="checkbox" :checked="!!draft[f.key]"
                     @change="setVal(f.key, $event.target.checked)" />
            </template>
            <template v-else-if="f.kind === 'choice'">
              <select :value="draft[f.key]" @change="setVal(f.key, $event.target.value)">
                <option v-for="c in f.choices" :key="c" :value="c">{{ c }}</option>
              </select>
            </template>
            <template v-else-if="f.kind === 'extensions'">
              <input class="grow" :value="(draft[f.key] || []).join(',')"
                     @change="setVal(f.key, $event.target.value.split(','))" />
            </template>
            <template v-else>
              <input class="num" :type="f.kind === 'int' || f.kind === 'float' ? 'number' : 'text'"
                     :step="f.kind === 'float' ? '0.01' : '1'"
                     :value="draft[f.key]" @change="setVal(f.key, $event.target.value)" />
            </template>
            <span class="range" v-if="f.min !== null && f.max !== null">{{ f.min }}~{{ f.max }}</span>
            <span class="cli" v-if="f.cli">{{ f.cli }}</span>
          </div>
        </div>
      </div>
    </div>
    <div class="clirow">
      <label class="lab">{{ t('params.cli_op', '等价 CLI（当前参数）') }}</label>
      <select v-model="cliOp">
        <option value="build">build</option>
        <option value="add">add</option>
        <option value="build-tiles">build-tiles</option>
        <option value="search">search</option>
        <option value="stats">stats</option>
        <option value="compact">compact</option>
      </select>
      <code class="grow">{{ state.cliText }}</code>
      <button @click="navigator.clipboard.writeText(state.cliText)">{{ t('btn.copy', '复制路径') }}</button>
    </div>
    <p class="hint">{{ (state.schema && state.schema.hint) || '' }}</p>
  </div>
</template>
