// ImageSearchTool · 图库检索管理器 — 前端自检钩子（P1 Gate 5 用）
// Copyright (C) 2026 zccored
// 本程序是自由软件：AGPL-3.0-only；完整条款见仓库 LICENSE。
//
// 用途：让 `devtools/verify_web_gui.py` 用**真实前端**测首屏（一屏缩略图）与端到端一致性，
// 而不是靠截图/肉眼。流程：scan → （可选）build → search(top_k=N) → 等 N 张缩略图加载完
// → 把耗时/命中数/缩略图成败/阶段边界回调给 Python（`api.selftest_report`）。
// 只在带 `?selftest=1` 时安装，正常使用不产生任何副作用。

import { onEvent } from './bridge.js';
import { nextTick } from 'vue';
import { applyTheme, currentTheme } from './theme.js';
import {
  state, cmd, setLocation, dedupScan, selectKeepBest, selectedPaths,
  openCompare, closeCompare
} from './store.js';

function waitFor(cond, timeoutMs, stepMs = 30) {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const tick = () => {
      if (cond()) { resolve(performance.now() - t0); return; }
      if (performance.now() - t0 > timeoutMs) { reject(new Error('等待超时')); return; }
      setTimeout(tick, stepMs);
    };
    tick();
  });
}

export function installSelftest() {
  // 始终安装（只在被显式调用时执行）：`devtools/verify_web_gui.py` 用它跑 Gate 5；
  // 正常使用时没有任何副作用（不会自动跑）。
  installSearchHook();
  installDedupHook();
  installCompareHook();
  installThemeHook();
}

/** Gate 6：亮/暗色切换（切主题 + 回报实际生效的背景色，供 Python 断言"两套配色都真的生效"）。 */
function installThemeHook() {
  window.__ise_selftest_theme = (name) => {
    if (name === 'light' || name === 'dark') applyTheme(name);
    return {
      theme: currentTheme(),
      attr: document.documentElement.getAttribute('data-theme') || '',
      bodyBg: getComputedStyle(document.body).backgroundColor,
      bodyFg: getComputedStyle(document.body).color
    };
  };
}

function installSearchHook() {
  window.__ise_selftest = async (opts = {}) => {
    const metrics = {
      ok: false, scan: 0, build: 0, searchMs: 0, hits: 0, topK: opts.topK || 60,
      thumbsOk: 0, thumbsFailed: 0, firstPaintMs: 0, phaseBoundaries: [],
      gallery: opts.gallery || state.dir, error: ''
    };
    const off = onEvent((ev) => {
      if (ev.event === 'phase_boundary') metrics.phaseBoundaries.push(ev.phase);
    });
    try {
      if (opts.gallery && opts.gallery !== state.dir) {
        state.dir = opts.gallery;
        await setLocation(opts.gallery, null);
      }
      state.page = 'search';            // 结果网格必须已挂载（占位不丢格判定依赖它）
      let t0 = performance.now();
      await cmd.scan(false);
      metrics.scan = Math.round(performance.now() - t0);

      if (opts.rebuild || !state.status.full_exists) {
        t0 = performance.now();
        await cmd.build(state.scan.paths, true, false);
        metrics.build = Math.round(performance.now() - t0);
      }
      const query = opts.query || state.query || (state.scan.paths[0] || '');
      t0 = performance.now();
      await cmd.search(query, opts.mode || 'full', null, metrics.topK, false);
      metrics.searchMs = Math.round(performance.now() - t0);
      metrics.hits = state.hits.length;

      // 首屏：等这一屏的缩略图落定（含 202 重试）；同时记录“前 10 张可见”时间
      const expect = Math.min(metrics.topK, state.hits.length);
      t0 = performance.now();
      await waitFor(() => {
        const imgs = Array.from(document.querySelectorAll('.grid .cell img'));
        if (imgs.length < expect) return false;
        const settled = imgs.filter((im) => im.complete);
        metrics.thumbsOk = settled.filter((im) => im.naturalWidth > 0).length;
        metrics.thumbsFailed = settled.filter((im) => im.naturalWidth === 0).length;
        const visible = settled.filter((im) => im.naturalWidth > 0).length;
        if (!metrics.firstVisibleMs && visible >= Math.min(10, expect)) {
          metrics.firstVisibleMs = Math.round(performance.now() - t0);
        }
        return settled.length >= expect;
      }, opts.timeoutMs || 30000);
      metrics.firstPaintMs = Math.round(performance.now() - t0);
      // 落定后再给“重试中”的图一次机会（占位会在这段时间内替换为缩略图）
      try {
        await waitFor(() => Array.from(document.querySelectorAll('.grid .cell img'))
          .every((im) => !im.complete || im.naturalWidth > 0), 6000, 100);
      } catch (e) { /* 超时不报错：下面统计真实失败数 */ }
      const final = Array.from(document.querySelectorAll('.grid .cell img'));
      metrics.thumbsOk = final.filter((im) => im.naturalWidth > 0).length;
      metrics.thumbsFailed = final.filter((im) => im.complete && im.naturalWidth === 0).length;
      // 失败样本（含重试后缀的 URL），便于离线定位
      metrics.failedSamples = final
        .filter((im) => im.complete && im.naturalWidth === 0)
        .slice(0, 6)
        .map((im) => im.currentSrc.slice(-72));
      metrics.ok = true;
    } catch (e) {
      metrics.error = String((e && e.message) || e);
    } finally {
      off();
      try { await cmd.report(metrics); } catch (e) { console.error(e); }
    }
    return metrics;
  };
}

function installDedupHook() {
  window.__ise_selftest_dedup = async (opts = {}) => {
    const m = {
      name: 'dedup', ok: false, scanned: 0, scanMs: 0, groups: 0, members: 0,
      rows: 0, firstScreenMs: 0, thumbsOk: 0, reopenMs: 0, reopenRows: 0,
      selCount: 0, error: ''
    };
    try {
      state.page = 'dedup';
      if (opts.gallery && opts.gallery !== state.dir) {
        state.dir = opts.gallery;
        await setLocation(opts.gallery, null);
      }
      if (!state.scan.paths.length) await cmd.scan(false);
      m.scanned = state.scan.paths.length;

      let t0 = performance.now();
      await dedupScan(state.scan.paths,
                      opts.threshold || state.dedup.threshold, state.prefix);
      m.scanMs = Math.round(performance.now() - t0);
      m.groups = (state.dedup.report?.groups || []).length;
      m.members = state.dedup.report?.n_images || 0;

      await nextTick();
      await waitFor(() => document.querySelectorAll('.vlist .vrow').length > 0, 15000);
      m.rows = document.querySelectorAll('.vlist .vrow').length;

      const want = Math.min(opts.firstScreen || 12, m.members);
      t0 = performance.now();
      await waitFor(() => Array.from(document.querySelectorAll('.vlist img.t96'))
        .filter((im) => im.complete && im.naturalWidth > 0).length >= want, 30000);
      m.firstScreenMs = Math.round(performance.now() - t0);
      const t96 = Array.from(document.querySelectorAll('.vlist img.t96'));
      m.thumbsOk = t96.filter((im) => im.naturalWidth > 0).length;

      selectKeepBest();
      m.selCount = selectedPaths().length;

      // 二次打开：切走再回来（报告在内存里，不重扫）
      state.page = 'search';
      await nextTick();
      t0 = performance.now();
      state.page = 'dedup';
      await nextTick();
      await waitFor(() => document.querySelectorAll('.vlist .vrow').length > 0, 5000);
      m.reopenMs = Math.round(performance.now() - t0);
      m.reopenRows = document.querySelectorAll('.vlist .vrow').length;
      m.ok = true;
    } catch (e) {
      m.error = String((e && e.message) || e);
    } finally {
      try { await cmd.report(m); } catch (e) { console.error(e); }
    }
    return m;
  };
}

function installCompareHook() {
  window.__ise_selftest_compare = async (opts = {}) => {
    const m = {
      name: 'compare', ok: false, opened: false, members: 0, panesRendered: false,
      leftImg: false, rightImg: false, zoomChanged: false, panChanged: false,
      stripDragOk: false, paneSwap: false, fullscreen: false, escClosed: false,
      error: ''
    };
    try {
      m.opened = openCompare(opts.groupIndex || 0, '');
      if (!m.opened) throw new Error('没有可对比的重复组');
      await nextTick();
      const panes = Array.from(document.querySelectorAll('.overlay .pane'));
      const imgs = Array.from(document.querySelectorAll('.overlay .paneimg img'));
      m.panesRendered = panes.length === 2 && imgs.length === 2;
      m.members = (state.compare.members || []).length;
      try {
        await waitFor(() => imgs.length === 2 && imgs.every((im) => im.complete
          && im.naturalWidth > 0), 45000);
      } catch (e) { /* 原图很大时可能超时：下面按实际值报告 */ }
      m.leftImg = !!(imgs[0] && imgs[0].naturalWidth > 0);
      m.rightImg = !!(imgs[1] && imgs[1].naturalWidth > 0);
      m.leftSrc = (imgs[0] && imgs[0].currentSrc) || '';
      m.rightSrc = (imgs[1] && imgs[1].currentSrc) || '';
      // 诊断：直接用 fetch 打同一 URL，区分“传输失败”与“<img> 未加载”
      const probe = async (u) => {
        if (!u) return 'no-src';
        try { const r = await fetch(u, { cache: 'no-store' }); return r.status; }
        catch (e) { return 'ERR:' + String(e && e.message); }
      };
      m.leftStatus = await probe(state.compare.open ? m.leftSrc : m.leftSrc);
      m.rightStatus = await probe(m.rightSrc);
      m.leftComplete = !!(imgs[0] && imgs[0].complete);
      m.rightComplete = !!(imgs[1] && imgs[1].complete);

      // 滚轮缩放（左侧）
      const s0 = state.compare.view.left.s;
      panes[0].dispatchEvent(new WheelEvent('wheel', {
        bubbles: true, cancelable: true, deltaY: -120, clientX: 300, clientY: 300
      }));
      m.zoomChanged = state.compare.view.left.s > s0;

      // 拖动平移（左侧）
      const x0 = state.compare.view.left.x;
      panes[0].dispatchEvent(new PointerEvent('pointerdown', {
        bubbles: true, clientX: 300, clientY: 300
      }));
      panes[0].dispatchEvent(new PointerEvent('pointermove', {
        bubbles: true, clientX: 360, clientY: 330
      }));
      panes[0].dispatchEvent(new PointerEvent('pointerup', { bubbles: true }));
      m.panChanged = state.compare.view.left.x !== x0;

      // 拖放：成员条第 2 张 → 右区
      const items = Array.from(document.querySelectorAll('.overlay .stripitem'));
      const target = (state.compare.members[1] || {}).path;
      if (items[1] && target) {
        const dt = new DataTransfer();
        items[1].dispatchEvent(new DragEvent('dragstart', {
          bubbles: true, dataTransfer: dt
        }));
        m.stripDragOk = dt.getData('text/ise-path') === target;
        panes[1].dispatchEvent(new DragEvent('drop', {
          bubbles: true, cancelable: true, dataTransfer: dt
        }));
        m.paneSwap = state.compare.right === target;
      }

      // 全屏（允许被拒绝：只记录结果）
      try {
        await document.documentElement.requestFullscreen();
        state.compare.fullscreen = !!document.fullscreenElement;
      } catch (e) { /* 某些环境需要用户手势 */ }
      m.fullscreen = !!document.fullscreenElement;
      try { if (document.fullscreenElement) await document.exitFullscreen(); } catch (e) { /* ignore */ }

      // Esc 关闭
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      await nextTick();
      m.escClosed = !state.compare.open;
      m.ok = true;
    } catch (e) {
      m.error = String((e && e.message) || e);
      try { closeCompare(); } catch (e2) { /* ignore */ }
    } finally {
      try { await cmd.report(m); } catch (e) { console.error(e); }
    }
    return m;
  };
}
