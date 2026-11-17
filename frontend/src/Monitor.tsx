import { useEffect, useRef } from 'react'
import { CLASS_COLOR, fmtTime } from './theme'
import { FS, type TraceRefs } from './useEcgStream'

const WINDOW_S = 8
const TRACE = '#3ef0a1'

interface Props {
  trace: React.RefObject<TraceRefs>
  live: boolean
}

/** Bedside-monitor style strip chart. Draws straight from refs on every
 *  animation frame, so 25 updates/s never trigger a React render. */
export function Monitor({ trace, live }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const range = useRef({ lo: -1, hi: 1.5 })

  useEffect(() => {
    const el = canvas.current!
    const ctx = el.getContext('2d')!
    let raf = 0
    let w = 0
    let h = 0

    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      const r = el.getBoundingClientRect()
      w = r.width
      h = r.height
      el.width = Math.round(w * dpr)
      el.height = Math.round(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()

    const draw = () => {
      const t = trace.current
      const n = WINDOW_S * FS
      const end = t.head
      const start = end - n
      const plotTop = 34
      const plotBottom = h - 42
      const ph = plotBottom - plotTop

      // Auto-range on what's visible, eased so the trace doesn't jump.
      let mn = Infinity
      let mx = -Infinity
      for (let i = Math.max(0, start); i < end; i += 2) {
        const v = t.buffer[i % t.buffer.length]
        if (v < mn) mn = v
        if (v > mx) mx = v
      }
      if (Number.isFinite(mn)) {
        const pad = Math.max(0.25, (mx - mn) * 0.12)
        let lo = mn - pad
        let hi = mx + pad
        if (hi - lo < 1.5) {
          const c = (hi + lo) / 2
          lo = c - 0.75
          hi = c + 0.75
        }
        range.current.lo += (lo - range.current.lo) * 0.08
        range.current.hi += (hi - range.current.hi) * 0.08
      }
      const { lo, hi } = range.current
      const x = (s: number) => ((s - start) / n) * w
      const y = (v: number) => plotTop + (1 - (v - lo) / (hi - lo)) * ph

      ctx.clearRect(0, 0, w, h)

      // ECG paper grid: 0.2 s / 0.5 mV boxes, with 1 s emphasis.
      const firstTick = Math.ceil(start / (0.2 * FS)) * 0.2 * FS
      for (let s = firstTick; s <= end; s += 0.2 * FS) {
        const major = Math.round(s / FS / 0.2) % 5 === 0
        ctx.strokeStyle = major ? 'rgba(62,240,161,0.14)' : 'rgba(62,240,161,0.05)'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(Math.round(x(s)) + 0.5, plotTop)
        ctx.lineTo(Math.round(x(s)) + 0.5, plotBottom)
        ctx.stroke()
        if (major && s >= 0) {
          ctx.fillStyle = 'rgba(160,200,185,0.45)'
          ctx.font = '11px "IBM Plex Mono", monospace'
          ctx.fillText(fmtTime(s / FS), x(s) + 4, h - 4)
        }
      }
      for (let v = Math.ceil(lo / 0.5) * 0.5; v <= hi; v += 0.5) {
        ctx.strokeStyle = 'rgba(62,240,161,0.06)'
        ctx.beginPath()
        ctx.moveTo(0, Math.round(y(v)) + 0.5)
        ctx.lineTo(w, Math.round(y(v)) + 0.5)
        ctx.stroke()
      }

      // Trace with a phosphor glow.
      const step = Math.max(1, Math.floor(n / (w * 1.5)))
      ctx.save()
      ctx.strokeStyle = TRACE
      ctx.lineWidth = 1.6
      ctx.lineJoin = 'round'
      ctx.shadowColor = 'rgba(62,240,161,0.55)'
      ctx.shadowBlur = 8
      ctx.beginPath()
      let started = false
      for (let s = Math.max(0, start); s < end; s += step) {
        const px = x(s)
        const py = y(t.buffer[s % t.buffer.length])
        if (!started) {
          ctx.moveTo(px, py)
          started = true
        } else ctx.lineTo(px, py)
      }
      ctx.stroke()
      ctx.restore()

      if (started && end > 0) {
        const py = y(t.buffer[(end - 1) % t.buffer.length])
        ctx.fillStyle = TRACE
        ctx.shadowColor = TRACE
        ctx.shadowBlur = live ? 14 : 0
        ctx.beginPath()
        ctx.arc(w - 2, py, 3, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0
      }

      // Beat markers: model label on top, cardiologist label underneath.
      ctx.font = '600 11px "IBM Plex Mono", monospace'
      ctx.textAlign = 'center'
      for (const b of t.beats) {
        if (b.sample < start || b.sample > end) continue
        const bx = x(b.sample)
        const wrong = b.truth !== null && b.truth !== 'Q' && b.truth !== b.label
        const c = CLASS_COLOR[b.label]
        ctx.strokeStyle = b.label === 'N' ? 'rgba(127,216,176,0.25)' : c
        ctx.setLineDash(b.label === 'N' ? [2, 4] : [])
        ctx.beginPath()
        ctx.moveTo(bx + 0.5, 22)
        ctx.lineTo(bx + 0.5, plotBottom)
        ctx.stroke()
        ctx.setLineDash([])

        const label = b.label
        const pw = 18
        ctx.fillStyle = b.label === 'N' ? 'rgba(127,216,176,0.12)' : c
        roundRect(ctx, bx - pw / 2, 6, pw, 16, 4)
        ctx.fill()
        if (wrong) {
          ctx.strokeStyle = '#ffffff'
          ctx.lineWidth = 1.5
          roundRect(ctx, bx - pw / 2 - 2, 4, pw + 4, 20, 5)
          ctx.stroke()
          ctx.lineWidth = 1
        }
        ctx.fillStyle = b.label === 'N' ? '#7fd8b0' : '#0b1210'
        ctx.fillText(label, bx, 18)
        if (b.truth) {
          ctx.fillStyle = wrong ? '#ffffff' : 'rgba(160,200,185,0.5)'
          ctx.font = '500 10px "IBM Plex Mono", monospace'
          ctx.fillText(b.truth, bx, plotBottom + 12)
          ctx.font = '600 11px "IBM Plex Mono", monospace'
        }
      }
      ctx.textAlign = 'left'

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [trace, live])

  return <canvas ref={canvas} className="monitor-canvas" aria-label="Live ECG trace, lead MLII" role="img" />
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}
