// ImageSearchTool · 图库检索管理器 — 前端 ↔ Python 桥
// Copyright (C) 2026 zccored
//
// 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
// 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见仓库根 LICENSE。
// This program is free software under the GNU Affero General Public License
// v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
//
// 两条实现约束（都踩过坑，`devtools/verify_web_gui.py` 的 Gate 会验）：
//   1) 必须等 `pywebviewready` 再调用 window.pywebview.api.*（api 对象先存在、方法后挂载）；
//   2) 不得依赖 stdout：本文件只用事件与返回值沟通。
//
// 事件入口：window.__ise_event(events[])；事件形如
//   { event: 'log'|'progress'|'phase_boundary'|'viz_frame'|'perf_report'|'task_done'|'task_error',
//     task_id, ... }
// 长任务：`run(op, args)` —— js_api 立即返回 {accepted, task_id}，结果由 task_done/task_error
//   事件按 task_id 关联（Python 侧在独立线程执行，UI 永不阻塞）。

let _readyPromise = null;
const _listeners = new Set();
const _tasks = new Map(); // task_id -> { resolve, reject, op }
let _seq = 0;

export function onEvent(fn) {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

export function token() {
  return (window.pywebview && window.pywebview.token) || '';
}

export function ready() {
  if (!_readyPromise) {
    _readyPromise = new Promise((resolve) => {
      const done = () => { window.__iseReadyFired = true; resolve(true); };
      if (window.__iseReadyFired) { done(); return; }
      window.addEventListener('pywebviewready', done, { once: true });
    });
  }
  return _readyPromise;
}

export async function call(method, ...args) {
  await ready();
  const api = window.pywebview && window.pywebview.api;
  const fn = api && api[method];
  if (typeof fn !== 'function') {
    throw new Error(`api.${method} 尚未就绪`);
  }
  return fn(...args, token()); // pythonnet 侧按位置调用：token 永远放最后
}

/** 注册事件入口并通知 Python 侧“可以开始推送”（早于它会丢事件）。 */
export async function boot(handler) {
  window.__ise_event = (events) => {
    for (const ev of events || []) dispatch(ev);
  };
  if (handler) onEvent(handler);
  await ready();
  try { await call('ready'); } catch (e) { console.error('ready 失败', e); }
  return true;
}

function dispatch(ev) {
  const kind = ev && ev.event;
  if (kind === 'task_done' || kind === 'task_error') {
    const t = _tasks.get(ev.task_id);
    if (t) {
      _tasks.delete(ev.task_id);
      if (kind === 'task_done') t.resolve(ev.result);
      else t.reject(new Error(ev.error || '任务失败'));
    }
  }
  for (const fn of _listeners) {
    try { fn(ev); } catch (e) { console.error('事件处理异常', e); }
  }
}

/**
 * 跑一个长任务命令。`op` 同时是 js_api 方法名（scan/build_index/add_index/
 * tiles_index/compact/search/dedup_scan/stats），`args` 按各方法的位置参数顺序给。
 */
export async function run(op, args = [], taskId = '') {
  const tid = taskId || `${op}-${++_seq}-${Date.now()}`;
  const done = new Promise((resolve, reject) => {
    _tasks.set(tid, { resolve, reject, op });
  });
  await call(op, tid, ...args);
  return done;
}

export function apiCall(method, ...args) { // 短方法（同步返回）
  return call(method, ...args);
}
