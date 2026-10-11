export const NOT_EVALUATED = 'Not evaluated'

export function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  if (typeof v !== 'number') return String(v)
  if (Math.abs(v) >= 1e6) return v.toExponential(2)
  if (Number.isInteger(v)) return v.toLocaleString()
  return v.toLocaleString(undefined, { maximumFractionDigits: d })
}

export const metric = (v, d = 3) => (v === null || v === undefined ? NOT_EVALUATED : Number(v).toFixed(d))
export const pct = (v, d = 0) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`)

export function dt(s) {
  if (!s) return '—'
  try { return new Date(s).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) } catch { return s }
}

export const FIELD_LABELS = {
  crop: 'Crop', state: 'State', season: 'Season', n: 'Nitrogen (N)', p: 'Phosphorus (P)', k: 'Potassium (K)', ph: 'Soil pH',
  rainfall: 'Rainfall (mm)', temperature: 'Temperature (°C)', humidity: 'Humidity (%)', fertilizer: 'Fertilizer', pesticide: 'Pesticide',
}

export const PROVENANCE = {
  user_provided: { label: 'You provided', tone: 'green' },
  parsed_from_query: { label: 'Parsed from query', tone: 'blue' },
  model_inferred: { label: 'LLM inferred', tone: 'amber' },
  missing: { label: 'Missing', tone: 'neutral' },
}

export const RISK_TONE = { low: 'green', moderate: 'amber', high: 'red', insufficient_information: 'neutral' }
export const VERDICT_TONE = { supported: 'green', partially_supported: 'amber', unsupported: 'red', not_run: 'neutral' }
export const VERDICT_LABEL = { supported: 'Supported', partially_supported: 'Partially supported', unsupported: 'Unsupported', not_run: 'Not run' }

export const CHART = ['#c4f26a', '#34d399', '#7dd3a8', '#86b84a', '#2dd4bf', '#a3e635', '#fbbf24', '#5eead4', '#bef264', '#4ade80']
