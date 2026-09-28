<script setup>
// ImageSearchTool · 处理过程可视化（canvas）：粗筛 64×64 二值点阵 / 精排 16×16 RGB 象限
// Copyright (C) 2026 zccored · 本程序是自由软件：AGPL-3.0-only（见仓库根 LICENSE）
//
// 帧由 Python 侧节流后以 base64 送达（“最近一帧优先”，等价 tkinter 版的有界队列+丢中间帧）。

import { onMounted, ref, watch } from 'vue';

const props = defineProps({ frame: { type: Object, required: true } });
const cv = ref(null);

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}

function draw() {
  const el = cv.value;
  const f = props.frame;
  if (!el) return;
  const w = el.width;
  const h = el.height;
  const ctx = el.getContext('2d');
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, w, h);
  if (!f || !f.data || !f.w || !f.h) return;
  const bytes = b64ToBytes(f.data);
  const img = ctx.createImageData(f.w, f.h);
  if (f.kind === 'fine') {
    for (let i = 0, p = 0; i < f.w * f.h; i += 1, p += 3) {
      img.data[i * 4] = bytes[p];
      img.data[i * 4 + 1] = bytes[p + 1];
      img.data[i * 4 + 2] = bytes[p + 2];
      img.data[i * 4 + 3] = 255;
    }
  } else {
    for (let i = 0; i < f.w * f.h; i += 1) {
      const on = bytes[i] > 127;
      img.data[i * 4] = on ? 140 : 10;
      img.data[i * 4 + 1] = on ? 255 : 15;
      img.data[i * 4 + 2] = on ? 140 : 22;
      img.data[i * 4 + 3] = 255;
    }
  }
  const off = document.createElement('canvas');
  off.width = f.w; off.height = f.h;
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  const side = Math.max(1, Math.floor(Math.min(w, h) / Math.max(f.w, 1)));
  const dw = f.w * side;
  const dh = f.h * side;
  ctx.drawImage(off, Math.floor((w - dw) / 2), Math.floor((h - dh) / 2), dw, dh);
}

watch(() => props.frame.frame, draw);
onMounted(draw);
</script>

<template>
  <canvas ref="cv" class="vizcanvas" width="296" height="150" />
</template>
