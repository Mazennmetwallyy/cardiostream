import { useCallback, useEffect, useRef, useState } from 'react'
import { CLASSES, streamURL, type Beat, type BeatClass, type StreamMessage } from './api'

export const FS = 360
const BUFFER_S = 12

export type Status = 'idle' | 'connecting' | 'live' | 'paused' | 'ended' | 'error'

export interface Stats {
  predicted: Record<BeatClass, number>
  matched: number // beats with a cardiologist N/S/V/F label
  agree: number
  // per true class: [caught, total]
  recall: Record<BeatClass, [number, number]>
}

const emptyStats = (): Stats => ({
  predicted: { N: 0, S: 0, V: 0, F: 0 },
  matched: 0,
  agree: 0,
  recall: { N: [0, 0], S: [0, 0], V: [0, 0], F: [0, 0] },
})

/** Live trace data lives in refs (read by the canvas every animation frame);
 *  React state only changes once per heartbeat. */
export interface TraceRefs {
  buffer: Float32Array // ring buffer of millivolts
  head: number // global sample index one past the newest sample
  beats: Beat[] // recent classified beats (for markers)
}

export function useEcgStream() {
  const trace = useRef<TraceRefs>({ buffer: new Float32Array(BUFFER_S * FS), head: 0, beats: [] })
  const ws = useRef<WebSocket | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [stats, setStats] = useState<Stats>(emptyStats)
  const [lastBeat, setLastBeat] = useState<Beat | null>(null)
  const [events, setEvents] = useState<Beat[]>([])
  const [length, setLength] = useState(0)

  const close = useCallback(() => {
    const s = ws.current
    ws.current = null
    if (s) {
      s.onclose = null
      s.close()
    }
  }, [])

  const reset = useCallback(() => {
    close()
    trace.current = { buffer: new Float32Array(BUFFER_S * FS), head: 0, beats: [] }
    setStats(emptyStats())
    setLastBeat(null)
    setEvents([])
    setStatus('idle')
  }, [close])

  const onBeat = useCallback((b: Beat) => {
    const t = trace.current
    t.beats.push(b)
    const oldest = t.head - BUFFER_S * FS
    while (t.beats.length && t.beats[0].sample < oldest) t.beats.shift()
    setLastBeat(b)
    setStats((s) => {
      const n = { ...s, predicted: { ...s.predicted, [b.label]: s.predicted[b.label] + 1 }, recall: { ...s.recall } }
      if (b.truth && b.truth !== 'Q') {
        n.matched += 1
        if (b.truth === b.label) n.agree += 1
        const [c, tot] = s.recall[b.truth]
        n.recall[b.truth] = [c + (b.label === b.truth ? 1 : 0), tot + 1]
      }
      return n
    })
    if (b.label !== 'N' || (b.truth && b.truth !== 'Q' && b.truth !== b.label)) {
      setEvents((e) => [b, ...e].slice(0, 60))
    }
  }, [])

  const play = useCallback(
    (record: string, speed: number, fromStart: boolean) => {
      close()
      const t = trace.current
      const startS = fromStart ? 0 : t.head / FS
      setStatus('connecting')
      const sock = new WebSocket(streamURL(record, speed, startS))
      ws.current = sock
      sock.onmessage = (ev) => {
        const m = JSON.parse(ev.data) as StreamMessage
        if (m.t === 'meta') {
          setLength(m.length)
          t.head = m.start
          setStatus('live')
        } else if (m.t === 'samples') {
          const buf = t.buffer
          for (let i = 0; i < m.mv.length; i++) buf[(m.start + i) % buf.length] = m.mv[i]
          t.head = m.start + m.mv.length
        } else if (m.t === 'beat') {
          onBeat(m)
        } else if (m.t === 'end') {
          setStatus('ended')
        }
      }
      sock.onerror = () => setStatus('error')
      sock.onclose = (ev) => {
        if (ws.current === sock) setStatus(ev.code === 1000 ? 'ended' : 'error')
      }
    },
    [close, onBeat],
  )

  const pause = useCallback(() => {
    close()
    setStatus('paused')
  }, [close])

  useEffect(() => close, [close])

  return { trace, status, stats, lastBeat, events, length, play, pause, reset }
}

export const classes = CLASSES
