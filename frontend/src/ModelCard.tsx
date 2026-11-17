import { CLASSES, CLASS_INFO, type EvalReport, type ModelCard as Card } from './api'
import { CLASS_COLOR } from './theme'

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export function ModelCard({ card }: { card: Card | null }) {
  if (!card) return <div className="panel muted">Loading model card…</div>
  const s = card.test_streaming
  const q = s.qrs_detection
  return (
    <div className="card-grid">
      <section className="panel span-2">
        <h2>How it was evaluated</h2>
        <p className="lede">
          Trained on {card.split.train_records.length} patients for a fixed schedule (no tuning against test
          results), then tested once on <strong>{card.split.test_records.length} patients it never saw</strong> (the
          inter-patient protocol of de Chazal et al., 2004). Most published MIT-BIH results mix beats from the
          same patients into training and test sets, which inflates accuracy to ~99%. This page reports the
          harder, honest number.
        </p>
        <div className="kpis">
          <Kpi label="QRS detection sensitivity" value={pct(q.sensitivity)} note={`${q.reference_beats.toLocaleString()} reference beats`} />
          <Kpi label="QRS detection precision" value={pct(q.ppv)} note="detected beats that are real" />
          <Kpi label="Beat accuracy (live pipeline)" value={pct(s.accuracy)} note="detector → features → model" />
          <Kpi label="Macro F1 (live pipeline)" value={s.macro_f1.toFixed(3)} note="all four classes weighted equally" />
        </div>
      </section>

      <section className="panel">
        <h2>Per-class results · live pipeline</h2>
        <ClassTable report={s} />
        <p className="footnote">
          Same metrics using reference R-peaks instead of the detector: accuracy{' '}
          {pct(card.test_reference_peaks.accuracy)}, macro F1 {card.test_reference_peaks.macro_f1.toFixed(3)}.
        </p>
      </section>

      <section className="panel">
        <h2>Confusion matrix · live pipeline</h2>
        <Confusion m={s.confusion_matrix} />
      </section>

      <section className="panel">
        <h2>Training</h2>
        <TrainingCurve history={card.training.history} />
        <dl className="facts">
          <dt>Model</dt>
          <dd>{card.model}</dd>
          <dt>Parameters</dt>
          <dd>{card.parameters.toLocaleString()}</dd>
          <dt>Schedule</dt>
          <dd>{card.training.epochs} epochs, one-cycle LR, final weights kept</dd>
          <dt>Training time</dt>
          <dd>
            {Math.max(1, Math.round(card.training.seconds / 60))} min on a laptop ({card.training.hardware})
          </dd>
          <dt>Serving</dt>
          <dd>ONNX Runtime, no PyTorch in the server image</dd>
        </dl>
      </section>

      <section className="panel">
        <h2>Limitations</h2>
        <ul className="limits">
          <li>
            Supraventricular (S) beats are the hardest class across patients: they look like normal beats and are
            told apart mostly by timing.
          </li>
          <li>Fusion (F) beats are rare (under 1% of training beats), so their scores are unstable.</li>
          <li>Single lead (MLII); 44 recordings (the 4 paced ones are excluded) made at one hospital lab, 1975–79.</li>
          <li>
            <strong>Research demo, not a medical device.</strong> It is not validated for clinical use.
          </li>
        </ul>
      </section>
    </div>
  )
}

function Kpi({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="kpi">
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      <div className="kpi-note">{note}</div>
    </div>
  )
}

function ClassTable({ report }: { report: EvalReport }) {
  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>Class</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>Beats</th>
        </tr>
      </thead>
      <tbody>
        {CLASSES.map((c) => {
          const r = report.per_class[c]
          return (
            <tr key={c}>
              <td>
                <span className="dot" style={{ background: CLASS_COLOR[c] }} />
                <span title={CLASS_INFO[c].short}>
                  {c} · {CLASS_INFO[c].name}
                </span>
              </td>
              <td>{pct(r.precision)}</td>
              <td>{pct(r.recall)}</td>
              <td>{r.f1.toFixed(3)}</td>
              <td>{r.support.toLocaleString()}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function Confusion({ m }: { m: number[][] }) {
  return (
    <div className="confusion">
      <div className="cm-corner">true ↓ / predicted →</div>
      {CLASSES.map((c) => (
        <div key={c} className="cm-head">
          {c}
        </div>
      ))}
      {m.map((row, i) => {
        const total = row.reduce((a, b) => a + b, 0) || 1
        return [
          <div key={`h${i}`} className="cm-head">
            {CLASSES[i]}
          </div>,
          ...row.map((v, j) => {
            const share = v / total
            const color = i === j ? '62,240,161' : '255,84,112'
            return (
              <div
                key={`${i}-${j}`}
                className="cm-cell"
                style={{ background: `rgba(${color},${0.06 + share * 0.6})` }}
                title={`${v.toLocaleString()} true ${CLASSES[i]} beats predicted as ${CLASSES[j]} (${pct(share)} of row)`}
              >
                <span>{v.toLocaleString()}</span>
                <small>{pct(share)}</small>
              </div>
            )
          }),
        ]
      })}
    </div>
  )
}

function TrainingCurve({ history }: { history: Card['training']['history'] }) {
  const W = 600
  const H = 120
  const pts = history.map((h, i) => [
    (i / Math.max(1, history.length - 1)) * (W - 20) + 10,
    H - 16 - h.holdout_macro_f1 * (H - 30),
  ])
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="curve" role="img" aria-label="Held-out macro F1 by epoch">
      <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke="#3ef0a1" strokeWidth="2" />
      <text x="10" y={H - 2} className="curve-label">
        epoch 1
      </text>
      <text x={W - 10} y={H - 2} textAnchor="end" className="curve-label">
        {history.length}
      </text>
      <text x="10" y="12" className="curve-label">
        macro F1 on 10% held-out training beats (0 to 1)
      </text>
    </svg>
  )
}
