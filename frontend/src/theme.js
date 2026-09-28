// ImageSearchTool · 图库检索管理器 — 亮/暗色主题（只切 <html data-theme>，配色全在 theme.css）
// Copyright (C) 2026 zccored
// 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
// This program is free software under the GNU Affero General Public License v3.0.
//
// 约定：
//   * **暗色是默认值**（写在 theme.css 的 `:root` 里），亮色由 `:root[data-theme="light"]` 覆盖；
//   * 选择存在 `localStorage['ise.theme']`，下次启动沿用（`index.html` 里有一份等价的内联逻辑，
//     用于在首屏 CSS 生效前恢复，避免闪一下默认色）；
//   * 外置 `ui_theme.css` 仍可覆盖**两套**配色：写 `:root { … }` 改暗色，
//     写 `:root[data-theme="light"] { … }` 改亮色（后者更具体，不会被前者覆盖）。

const KEY = 'ise.theme';
export const THEMES = ['dark', 'light'];

export function currentTheme() {
  const v = document.documentElement.dataset.theme;
  return THEMES.includes(v) ? v : 'dark';
}

export function applyTheme(name) {
  const v = THEMES.includes(name) ? name : 'dark';
  document.documentElement.dataset.theme = v;
  try {
    localStorage.setItem(KEY, v);
  } catch (e) { /* 隐私模式 / 存储被禁：忽略，仅本次生效 */ }
  return v;
}

/** 启动时恢复上次选择（幂等；与 index.html 的内联片段等价）。 */
export function initTheme() {
  let saved = '';
  try {
    saved = localStorage.getItem(KEY) || '';
  } catch (e) { saved = ''; }
  if (THEMES.includes(saved)) document.documentElement.dataset.theme = saved;
}

/** 在亮/暗之间切换，返回切换后的主题名。 */
export function toggleTheme() {
  return applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
}
