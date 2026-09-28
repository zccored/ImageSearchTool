// ImageSearchTool · 图库检索管理器 — 前端共享状态与命令（唯一入口：js_api）
// Copyright (C) 2026 zccored
// 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
// This program is free software under the GNU Affero General Public License v3.0.
//
// 设计约定：
//   * 不引入状态管理库：reactive 单例 + 事件 reducer（bridge.onEvent 汇入）；
//   * **默认值不在这里**：参数页一律从 `get_config_schema()` 拉（唯一出处 Config）；
//   * 网格规则 `index === 命中下标`：占位不丢格（缩略图失败也占住这一格）。

import { reactive } from 'vue';
import { apiCall, run } from './bridge.js';

export const state = reactive({
  ready: false,
  strings: {},                 // ui_strings.json 外部覆盖（见 loadStrings）
  dir: '',
  prefix: '',
  location: { prefix: '', tiles_prefix: '', root: '' },
  status: { full_exists: false, tiles_exists: false, n: 0, tiles_n: 0, storage: '' },
  page: 'list',
  busy: false,
  statusText: '就绪：先选图库目录 → 扫描 → 建索引 → 查询',
  logs: [],                    // {time, level, text}
  logLimit: 800,
  progress: {},                // task_id -> {done,total,phase,boundary}
  viz: { kind: '', w: 0, h: 0, data: null, frame: 0 },
  perf: [],                    // {path, modal, stage, ts}
  schema: null,
  config: {},
  cliText: '',
  scan: { count: 0, broken: 0, elapsed: 0, formats: {}, paths: [] },
  indexed: [],                 // 已入库路径（normcase）
  selected: [],                // 列表里勾选的路径
  query: '',
  mode: 'full',
  hits: [],
  result: null,
  selIndex: -1,
  // ---- 去重审查（P2）----
  dedup: {
    report: null,            // 加工后的 DupReport（组/成员已带 wasted_bytes/thumb）
    threshold: 2.0,          // 汉明比例上限（%），与 tkinter 版同一输入口径
    sync: true,              // 删除/移动后同步索引（prune）
    sel: {},                 // path -> 是否勾选（= 将被删除/移动）
    filter: '',              // 只看含勾选的组 / 按文件名过滤
    onlySelected: false,
    busy: false,
    last: null               // 最近一次 删除/移动 结果
  },
  // ---- 大图对比（P2：页内 overlay + Fullscreen API）----
  compare: {
    open: false,
    groupId: -1,
    members: [],             // [{path,name,size,thumb,indexed,exact_copy}]
    left: '', right: '',
    active: 'left',
    fullscreen: false,
    view: { left: { s: 1, x: 0, y: 0 }, right: { s: 1, x: 0, y: 0 } }
  },
  errors: []
});

// ---------------------------------------------------------------- 文案
export function t(key, fallback) {
  const v = state.strings && state.strings[key];
  return (v === undefined || v === null || v === '') ? fallback : v;
}

/** 外置文案：`/ui_strings.json`（随包分发，优先于内置默认）；缺失时静默回落。 */
export async function loadStrings() {
  try {
    const r = await fetch('/ui_strings.json', { cache: 'no-store' });
    if (r.ok) state.strings = await r.json();
  } catch (e) { /* 没有外置文案：用内置默认 */ }
}

// ---------------------------------------------------------------- 事件 reducer
function applyDone(ev) {
  const op = ev.op;
  const r = ev.result || {};
  if (op === 'scan') {
    state.scan = {
      count: r.count || 0, broken: r.broken || 0, elapsed: r.elapsed || 0,
      formats: r.formats || {}, paths: r.paths || []
    };
    state.statusText = `扫描完成：${r.count} 张`
      + (r.broken ? `（跳过损坏 ${r.broken}）` : '');
  } else if (op === 'build' || op === 'add' || op === 'tiles') {
    state.statusText = `${r.title || op}完成：${r.n}`;
  } else if (op === 'compact') {
    state.statusText = '索引存储已优化（下次加载更快、内存更省）';
  } else if (op === 'search') {
    state.hits = r.hits || [];
    state.result = r;
    state.selIndex = state.hits.length ? 0 : -1;
    state.statusText = `检索完成，返回 ${state.hits.length} 个结果`;
  } else if (op === 'stats') {
    state.statusText = `索引统计：${r.n} 条`;
  } else if (op === 'dedup_scan') {
    state.dedup.report = r;
    state.dedup.sel = Object.assign({}, state.dedup.sel);
    state.dedup.busy = false;
    const g = (r.groups || []).length;
    state.statusText = `查验去重：${r.scanned} 张，${g} 组重复`
      + `（完全 ${r.n_exact} / 近似 ${r.n_near}），可释放 ${humanBytes(r.wasted_bytes)}`;
  } else if (op === 'dedup_apply') {
    state.dedup.last = r;
    state.dedup.busy = false;
    const n = (r.removed || []).length;
    const bad = (r.failed || []).length;
    state.statusText = `${r.dest ? '移动' : '删除'}完成：${n} 张`
      + (bad ? `（失败 ${bad}）` : '');
    dropRemoved(r.removed || []);
    refreshStatus();
  }
  if (op !== 'search' && op !== 'dedup_scan' && op !== 'dedup_apply') refreshStatus();
  setBusy(false);
}

/** 字节数 -> 人类可读（与 Python 侧 human_bytes 同量级，仅显示用）。 */
export function humanBytes(n) {
  const v = Number(n) || 0;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let x = Math.abs(v);
  while (x >= 1024 && i < units.length - 1) { x /= 1024; i += 1; }
  return `${x.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/** 删除/移动成功后：从报告里摘掉这些路径（组不足 2 张则删除该组）。 */
function dropRemoved(removed) {
  const rm = new Set(removed.map((p) => p));
  const rep = state.dedup.report;
  if (!rep) return;
  for (const g of rep.groups) {
    g.members = (g.members || []).filter((m) => !rm.has(m.path));
    g.n_members = g.members.length;
    g.wasted_bytes = g.members.slice(1).reduce((a, m) => a + (m.size || 0), 0);
  }
  rep.groups = rep.groups.filter((g) => g.members.length >= 2);
  rep.n_exact = rep.groups.filter((g) => g.all_exact).length;
  rep.n_near = rep.groups.filter((g) => !g.all_exact).length;
  rep.n_images = rep.groups.reduce((a, g) => a + g.n_members, 0);
  rep.wasted_bytes = rep.groups.reduce((a, g) => a + g.wasted_bytes, 0);
  state.indexed = state.indexed.filter((p) => !removed.some(
    (x) => normPath(x) === p));
  if (state.compare.open
      && state.compare.members.some((m) => rm.has(m.path))) {
    closeCompare();
  }
}

export function normPath(p) {
  const s = String(p || '').replace(/\//g, '\\');
  return /^[a-z]:\\/i.test(s) ? s.toLowerCase() : s;
}

export function reduce(ev) {
  switch (ev.event) {
    case 'log':
      state.logs.push({ time: ev.time, level: ev.level, text: ev.text });
      if (state.logs.length > state.logLimit) {
        state.logs.splice(0, state.logs.length - state.logLimit);
      }
      break;
    case 'progress':
      state.progress[ev.task_id] = {
        done: ev.done, total: ev.total, phase: ev.phase,
        boundary: (state.progress[ev.task_id] || {}).boundary
      };
      break;
    case 'phase_boundary':
      state.progress[ev.task_id] = Object.assign(
        {}, state.progress[ev.task_id], { boundary: ev.phase });
      break;
    case 'viz_frame':
      state.viz = { kind: ev.kind, w: ev.w, h: ev.h, data: ev.b64,
                    frame: state.viz.frame + 1 };
      break;
    case 'perf_report':
      state.perf.push({ path: ev.path, modal: ev.modal, stage: ev.stage, ts: Date.now() });
      break;
    case 'task_done':
      applyDone(ev);
      break;
    case 'task_error':
      state.errors.push({ op: ev.op, error: ev.error });
      state.logs.push({ time: '', level: 'ERROR', text: `❌ ${ev.op} 失败：${ev.error}` });
      state.statusText = `<${ev.op}> 失败：${ev.error}`;
      setBusy(false);
      break;
    default:
      break;
  }
}

export function setBusy(on) {
  state.busy = !!on;
}

// ---------------------------------------------------------------- 命令
export async function refreshStatus() {
  try {
    state.status = await apiCall('index_status', state.prefix || null);
    state.prefix = state.status.prefix || state.prefix;
    const idx = await apiCall('indexed_paths', state.prefix || null);
    state.indexed = (idx && idx.paths) || [];
  } catch (e) {
    state.logs.push({ time: '', level: 'WARN', text: `状态刷新失败：${e.message}` });
  }
}

export async function setLocation(root, prefix) {
  const res = await apiCall('set_location', root || null, prefix || null);
  state.location = res.location;
  state.prefix = res.location.prefix;
  state.status = res.status;
  if (root) state.dir = res.location.root;
  return res;
}

export async function loadSchema() {
  state.schema = await apiCall('get_config_schema');
  state.config = await apiCall('get_config');
  return state.schema;
}

export async function applyConfig(values) {
  const res = await apiCall('set_config', values);
  state.config = await apiCall('get_config');
  return res;
}

export async function refreshCli(op = 'build') {
  try {
    const r = await apiCall('cli_command', op);
    state.cliText = r.command || '';
  } catch (e) { state.cliText = ''; }
}

/** 长任务统一入口：置忙 → 发命令 → 等 task_done/task_error（事件同时驱动进度/日志）。 */
export function task(op, args, label) {
  setBusy(true);
  state.statusText = `${label || op} …（操作期间请勿重复点击）`;
  return run(op, args).finally(() => setBusy(false));
}

export const cmd = {
  pickDir: () => apiCall('choose_directory', '选择图库目录'),
  pickFile: () => apiCall('choose_file', '选择查询图片'),
  pickSave: (name) => apiCall('choose_save_file', '保存总览图',
                              name || 'search_result.png'),
  scan: (verify) => task('scan', [state.dir, true, !!verify], '扫描图库'),
  build: (paths, force, perf) => task('build_index',
    [state.prefix || null, paths || null, null, force !== false, !!perf,
     '全部图片新建索引'], '新建索引'),
  add: (paths, perf) => task('add_index',
    [state.prefix || null, paths || null, null, !!perf, '增量入库（自动去重）'],
    '增量入库'),
  tiles: (paths, exists, perf) => task('tiles_index',
    [null, paths || null, null, exists, !!perf],
    exists ? '子图索引增量' : '子图索引构建'),
  compact: (perf) => task('compact', [null, !!perf], '优化索引存储'),
  search: (query, mode, coarseK, topK, perf) => task('search',
    [query, mode, state.prefix || null, coarseK || null, topK || null, !!perf],
    `以图搜图(${mode})`),
  dedup: (paths, threshold) => task('dedup_scan',
    [paths, threshold, state.prefix || null], '查验去重'),
  stats: () => task('stats', [state.prefix || null], '索引统计'),
  release: () => apiCall('release_engines', '界面手动释放'),
  openPath: (p) => apiCall('open_in_shell', p),
  exportSheet: (paths, out) => apiCall('export_sheet', paths, out),
  thumbUrl: (p) => apiCall('thumb_url', p),
  latestPerf: () => apiCall('latest_perf_report'),
  report: (metrics) => apiCall('selftest_report', metrics),
  imageUrl: (p) => apiCall('image_url', p),
  confirm: (title, message) => apiCall('confirm', title, message)
};

// ---------------------------------------------------------------- 去重命令（P2）
export async function dedupScan(paths, thresholdPct, prefix) {
  state.dedup.busy = true;
  state.statusText = '查验重复图：复用索引 + 解码未入库文件…';
  setBusy(true);
  try {
    const r = await run('dedup_scan',
      [paths || state.scan.paths, (Number(thresholdPct) || 2) / 100,
       prefix || state.prefix || null]);
    return r;
  } finally {
    state.dedup.busy = false;
    setBusy(false);
  }
}

export async function dedupApply(kind, paths, opts = {}) {
  if (!paths || !paths.length) throw new Error('没有勾选任何图片');
  state.dedup.busy = true;
  setBusy(true);
  state.statusText = `${kind === 'move' ? '移动' : '删除'} ${paths.length} 张…`;
  try {
    if (kind === 'move') {
      return await run('dedup_move',
        [paths, opts.dest || '', opts.baseRoot || state.dir || null,
         state.prefix || null, !!state.dedup.sync]);
    }
    return await run('dedup_delete',
      [paths, state.prefix || null, !!state.dedup.sync]);
  } finally {
    state.dedup.busy = false;
    setBusy(false);
  }
}

/** 勾选助手：保留每组第一张（报告里的“基准”），其余全选 —— 与 tkinter 版同口径。 */
export function selectKeepBest() {
  for (const g of (state.dedup.report?.groups || [])) {
    g.members.forEach((m, i) => { state.dedup.sel[m.path] = i > 0; });
  }
}

/** 只勾选“与保留项 MD5 完全相同、且尚未入库”的副本（最安全的清理目标）。 */
export function selectExactUnindexed() {
  const indexed = new Set(state.indexed);
  for (const g of (state.dedup.report?.groups || [])) {
    g.members.forEach((m, i) => {
      state.dedup.sel[m.path] = i > 0 && !!m.exact_copy
        && !indexed.has(normPath(m.path));
    });
  }
}

export function selectExact() {
  for (const g of (state.dedup.report?.groups || [])) {
    g.members.forEach((m, i) => { state.dedup.sel[m.path] = i > 0 && !!m.exact_copy; });
  }
}

export function selectNone() { state.dedup.sel = {}; }

export function invertSelection() {
  for (const g of (state.dedup.report?.groups || [])) {
    for (const m of (g.members || [])) {
      state.dedup.sel[m.path] = !state.dedup.sel[m.path];
    }
  }
  state.dedup.sel = Object.assign({}, state.dedup.sel);
}

export function toggleSel(path) {
  state.dedup.sel[path] = !state.dedup.sel[path];
  state.dedup.sel = Object.assign({}, state.dedup.sel);
}

export function selectedPaths() {
  const rep = state.dedup.report;
  if (!rep) return [];
  const out = [];
  for (const g of rep.groups) {
    for (const m of g.members) {
      if (state.dedup.sel[m.path]) out.push(m.path);
    }
  }
  return out;
}

export function selectedBytes() {
  const rep = state.dedup.report;
  let n = 0;
  for (const g of (rep?.groups || [])) {
    for (const m of g.members) if (state.dedup.sel[m.path]) n += m.size || 0;
  }
  return n;
}

// ---------------------------------------------------------------- 大图对比（P2）
export function openCompare(groupIndex, startPath = '') {
  const g = state.dedup.report?.groups?.[groupIndex];
  if (!g || (g.members || []).length < 2) return false;
  const members = g.members.map((m) => Object.assign({}, m));
  const start = startPath || members[0].path;
  const other = members.find((m) => m.path !== start) || members[0];
  Object.assign(state.compare, {
    open: true,
    groupId: g.gid,
    members,
    left: start,
    right: other.path,
    active: 'left',
    view: { left: { s: 1, x: 0, y: 0 }, right: { s: 1, x: 0, y: 0 } }
  });
  return true;
}

export function closeCompare() {
  state.compare.open = false;
  state.compare.members = [];
}

export function setSide(side, path) {
  if (side !== 'left' && side !== 'right') return;
  state.compare[side] = path;
  state.compare.view[side] = { s: 1, x: 0, y: 0 };
  state.compare.active = side;
}

export function swapSides() {
  const { left, right } = state.compare;
  state.compare.left = right;
  state.compare.right = left;
}

/** 把某侧的当前图拉到另一侧（对比窗里“拖到另一侧”的语义）。 */
export function moveToOther(side) {
  const other = side === 'left' ? 'right' : 'left';
  if (state.compare[side]) setSide(other, state.compare[side]);
}

export function zoomAt(side, factor, cx, cy) {
  const v = state.compare.view[side];
  if (!v) return;
  const s = Math.min(16, Math.max(0.05, v.s * factor));
  const k = s / v.s;
  v.x = cx - (cx - v.x) * k;
  v.y = cy - (cy - v.y) * k;
  v.s = s;
}

export function panBy(side, dx, dy) {
  const v = state.compare.view[side];
  if (!v) return;
  v.x += dx;
  v.y += dy;
}

export function fitSide(side) {
  state.compare.view[side] = { s: 1, x: 0, y: 0 };
}

export function resetView(side) { fitSide(side); }
