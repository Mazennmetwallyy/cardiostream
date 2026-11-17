export const CLASSES = ['N', 'S', 'V', 'F'] as const
export type BeatClass = (typeof CLASSES)[number]

export const CLASS_INFO: Record<BeatClass, { name: string; short: string }> = {
  N: { name: 'Normal', short: 'Normal or bundle branch block beat' },
  S: { name: 'Supraventricular', short: 'Premature beat starting above the ventricles (e.g. atrial)' },
  V: { name: 'Ventricular', short: 'Premature beat starting in the ventricles (PVC)' },
  F: { name: 'Fusion', short: 'Fusion of a ventricular and a normal beat' },
}

export interface RecordInfo {
  id: string
  rhythms: string[]
  duration_s: number
  beats: Record<BeatClass | 'Q', number>
}

export interface ClassReport {
  precision: number
  recall: number
  f1: number
  support: number
}

export interface EvalReport {
  accuracy: number
  macro_f1: number
  per_class: Record<BeatClass, ClassReport>
  confusion_matrix: number[][]
}

export interface ModelCard {
  model: string
  version: string
  parameters: number
  classes: BeatClass[]
  split: {
    protocol: string
    train_records: string[]
    test_records: string[]
    train_counts: Record<BeatClass, number>
    test_counts: Record<BeatClass, number>
  }
  training: {
    epochs: number
    seconds: number
    hardware: string
    class_weight_power?: number
    history: { epoch: number; train_loss: number; holdout_macro_f1: number }[]
  }
  test_reference_peaks: EvalReport
  test_streaming: EvalReport & {
    qrs_detection: { sensitivity: number; ppv: number; reference_beats: number }
  }
}

export type StreamMessage =
  | { t: 'meta'; record: string; fs: number; start: number; length: number }
  | { t: 'samples'; start: number; mv: number[] }
  | {
      t: 'beat'
      sample: number
      label: BeatClass
      probs: Record<BeatClass, number>
      rr: number
      hr: number
      truth: BeatClass | 'Q' | null
      pipeline_ms: number
    }
  | { t: 'end' }

export type Beat = Extract<StreamMessage, { t: 'beat' }>

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json() as Promise<T>
}

export const fetchRecords = () => getJSON<RecordInfo[]>('/api/records')
export const fetchModelCard = () => getJSON<ModelCard>('/api/model')

export function streamURL(record: string, speed: number, startSeconds: number): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const q = new URLSearchParams({ record, speed: String(speed), start: startSeconds.toFixed(2) })
  return `${proto}://${location.host}/api/stream?${q}`
}
