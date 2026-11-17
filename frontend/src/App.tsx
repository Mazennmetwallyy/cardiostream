import { useEffect, useState } from 'react'
import {
  CLASSES,
  CLASS_INFO,
  fetchModelCard,
  fetchRecords,
  type Beat,
  type ModelCard as Card,
  type RecordInfo,
} from './api'
import { ModelCard } from './ModelCard'
import { Monitor } from './Monitor'
import { CLASS_COLOR, fmtTime } from './theme'
import { FS, useEcgStream, type Stats } from './useEcgStream'

const REPO = 'https://github.com/Mazennmetwallyy/cardiostream'
const SPEEDS = [1, 2, 4, 8]
const DEFAULT_RECORD = '233'

export default function App() {
  const [tab, setTab] = useState<'live' | 'model'>('live')
  const [records, setRecords] = useState<RecordInfo[]>([])
  const [card, setCard] = useState<Card | null>(null)
  const [record, setRecord] = useState(DEFAULT_RECORD)
  const [speed, setSpeed] = useState(1)
  const [loadError, setLoadError] = useState(false)
  const s = useEcgStream()

  useEffect(() => {
    fetchRecords()
      .then((r) => {
        setRecords(r)
        s.play(DEFAULT_RECORD, 1, true)
      })
      .catch(() => setLoadError(true))
    fetchModelCard().then(setCard).catch(() => setLoadError(true))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const current = records.find((r) => r.id === record)
  const running = s.status === 'live' || s.status === 'connecting'

  const choose = (id: string) => {
    setRecord(id)
    s.reset()
    s.play(id, speed, true)
  }
  const changeSpeed = (v: number) => {
    setSpeed(v)
    if (running) s.play(record, v, false)
  }
  const toggle = () => (running ? s.pause() : s.play(record, speed, s.status === 'ended' || s.status === 'idle'))

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden>
            <path d="M3 18h6l2.5-5 3.5 12 4-19 3 12h7" fill="none" stroke="#3ef0a1" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div>
            <div className="brand-name">CardioStream</div>
            <div className="brand-sub">Real-time arrhythmia detection on streaming ECG</div>
          </div>
        </div>
        <nav className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'live'} className={tab === 'live' ? 'on' : ''} onClick={() => setTab('live')}>
            Live monitor
          </button>
          <button role="tab" aria-selected={tab === 'model'} className={tab === 'model' ? 'on' : ''} onClick={() => setTab('model')}>
            Model card
          </button>
          <a href={REPO} target="_blank" rel="noreferrer" className="gh">
            Source ↗
          </a>
        </nav>
      </header>

      {loadError && <div className="banner">Can't reach the API. Is the server running?</div>}
      {!loadError && records.length === 0 && (
        <div className="banner banner-info">
          Starting the server… The free host sleeps when nobody is using it, so the first visit can take up to a
          minute.
        </div>
      )}

      {tab === 'live' ? (
        <main className="live">
          <div className="controls">
            <label className="field">
              <span>Patient (held out from training)</span>
              <select value={record} onChange={(e) => choose(e.target.value)}>
                {records.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id}: {r.rhythms.slice(0, 2).join(', ') || 'unlabelled rhythm'}
                    {r.beats.V + r.beats.S + r.beats.F > 0 ? ` · ${r.beats.V} V, ${r.beats.S} S, ${r.beats.F} F` : ''}
                  </option>
                ))}
              </select>
            </label>
            <div className="field">
              <span>Speed</span>
              <div className="seg">
                {SPEEDS.map((v) => (
                  <button key={v} className={v === speed ? 'on' : ''} onClick={() => changeSpeed(v)}>
                    {v}×
                  </button>
                ))}
              </div>
            </div>
            <button className="play" onClick={toggle}>
              {running ? 'Pause' : s.status === 'ended' ? 'Replay' : 'Play'}
            </button>
            <StatusPill status={s.status} />
          </div>

          <section className="monitor panel">
            <div className="monitor-head">
              <span>Lead MLII · 360 Hz · patient {record}</span>
              <span className="legend">
                {CLASSES.map((c) => (
                  <span key={c}>
                    <i style={{ background: CLASS_COLOR[c] }} />
                    {c} {CLASS_INFO[c].name}
                  </span>
                ))}
                <span className="legend-note">top: model · bottom: cardiologist · white outline: disagreement</span>
              </span>
            </div>
            <Monitor trace={s.trace} live={s.status === 'live'} />
            <Progress head={s.trace.current.head} length={s.length} lastBeat={s.lastBeat} />
          </section>

          <aside className="vitals">
            <Vital label="Heart rate" value={s.lastBeat ? Math.round(s.lastBeat.hr).toString() : '--'} unit="bpm" />
            <Vital
              label="Agreement with cardiologists"
              value={s.stats.matched ? ((s.stats.agree / s.stats.matched) * 100).toFixed(1) : '--'}
              unit="%"
              note={`${s.stats.matched.toLocaleString()} beats so far`}
            />
            <LastBeat beat={s.lastBeat} />
          </aside>

          <section className="panel counts">
            <h2>This session</h2>
            <ClassCounts stats={s.stats} />
          </section>

          <section className="panel events">
            <h2>Event log</h2>
            <EventLog events={s.events} />
          </section>

          {current && (
            <p className="record-note">
              Record {current.id}: {fmtTime(current.duration_s)} of ECG ·{' '}
              {current.rhythms.join(', ') || 'no rhythm annotations'}. The model was never trained on this patient.
            </p>
          )}
        </main>
      ) : (
        <main>
          <ModelCard card={card} />
        </main>
      )}

      <footer className="foot">
        MIT-BIH Arrhythmia Database (Moody &amp; Mark, 2001), via PhysioNet. Research demo, not a medical device.
      </footer>
    </div>
  )
}

function StatusPill({ status }: { status: string }) {
  const text = { idle: 'Idle', connecting: 'Connecting', live: 'Live', paused: 'Paused', ended: 'Finished', error: 'Disconnected' }[status]
  return <span className={`pill pill-${status}`}>{text}</span>
}

function Progress({ head, length, lastBeat }: { head: number; length: number; lastBeat: Beat | null }) {
  const [, force] = useState(0)
  useEffect(() => {
    const id = setInterval(() => force((x) => x + 1), 500)
    return () => clearInterval(id)
  }, [])
  const frac = length ? Math.min(1, head / length) : 0
  return (
    <div className="progress">
      <div className="bar">
        <div style={{ width: `${frac * 100}%` }} />
      </div>
      <span>
        {fmtTime(head / FS)} / {fmtTime(length / FS)}
      </span>
      {lastBeat && <span className="lat">pipeline {lastBeat.pipeline_ms.toFixed(2)} ms per chunk</span>}
    </div>
  )
}

function Vital({ label, value, unit, note }: { label: string; value: string; unit: string; note?: string }) {
  return (
    <div className="panel vital">
      <div className="vital-label">{label}</div>
      <div className="vital-value">
        {value}
        <small>{unit}</small>
      </div>
      {note && <div className="vital-note">{note}</div>}
    </div>
  )
}

function LastBeat({ beat }: { beat: Beat | null }) {
  return (
    <div className="panel lastbeat">
      <div className="vital-label">Last beat · model confidence</div>
      {CLASSES.map((c) => {
        const p = beat?.probs[c] ?? 0
        return (
          <div key={c} className="prob">
            <span className="prob-k" style={{ color: CLASS_COLOR[c] }}>
              {c}
            </span>
            <div className="prob-bar">
              <div style={{ width: `${p * 100}%`, background: CLASS_COLOR[c] }} />
            </div>
            <span className="prob-v">{(p * 100).toFixed(1)}%</span>
          </div>
        )
      })}
      {beat && (
        <div className="vital-note">
          RR {Math.round(beat.rr * 1000)} ms · cardiologist label: {beat.truth ?? 'none'}
        </div>
      )}
    </div>
  )
}

function ClassCounts({ stats }: { stats: Stats }) {
  return (
    <div className="classcounts">
      {CLASSES.map((c) => {
        const [caught, total] = stats.recall[c]
        return (
          <div key={c} className="cc" style={{ borderColor: CLASS_COLOR[c] }}>
            <div className="cc-k" style={{ color: CLASS_COLOR[c] }}>
              {c}
            </div>
            <div className="cc-name">{CLASS_INFO[c].name}</div>
            <div className="cc-n">{stats.predicted[c].toLocaleString()}</div>
            <div className="cc-note">{total ? `caught ${caught} of ${total} labelled` : 'none labelled yet'}</div>
          </div>
        )
      })}
    </div>
  )
}

function EventLog({ events }: { events: Beat[] }) {
  if (!events.length) return <p className="muted">Abnormal beats and disagreements will appear here.</p>
  return (
    <ol className="log">
      {events.map((e) => {
        const wrong = e.truth && e.truth !== 'Q' && e.truth !== e.label
        return (
          <li key={e.sample} className={wrong ? 'wrong' : ''}>
            <span className="t">{fmtTime(e.sample / FS)}</span>
            <span className="k" style={{ color: CLASS_COLOR[e.label] }}>
              {e.label}
            </span>
            <span>
              {CLASS_INFO[e.label].name} · {(e.probs[e.label] * 100).toFixed(0)}%
            </span>
            <span className="truth">{wrong ? `cardiologist: ${e.truth}` : e.truth ? '✓ agrees' : ''}</span>
          </li>
        )
      })}
    </ol>
  )
}
