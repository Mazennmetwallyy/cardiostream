import type { BeatClass } from './api'

export const CLASS_COLOR: Record<BeatClass | 'Q', string> = {
  N: '#7fd8b0',
  S: '#ffc247',
  V: '#ff5470',
  F: '#b48cff',
  Q: '#8a9a94',
}

export function fmtTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${String(sec).padStart(2, '0')}`
}
