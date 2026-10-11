import { Badge, KV, Notice, ProgressBar } from './ui'
import { fmt } from '../utils/format'

export function CaseDetail({ item, record: rec0, text }) {
  const rec = rec0 || item?.record
  if (!rec) return null
  const comps = item?.priority_components
  return (
    <div className="space-y-3 text-sm">
      <KV items={[
        ['Case ID', rec.case_id], ['Crop', rec.crop], ['State', rec.state], ['Season', rec.season], ['Year', rec.year],
        ['Historical yield', fmt(rec.yield, 3)], ['Area (ha)', fmt(rec.area, 0)], ['Production', fmt(rec.production, 0)],
        ['Fertilizer (total)', fmt(rec.fertilizer, 0)], ['Pesticide (total)', fmt(rec.pesticide, 0)],
        ['Fertilizer / ha*', fmt(rec.fertilizer_per_ha, 1)], ['Pesticide / ha*', fmt(rec.pesticide_per_ha, 2)],
        ['N (state)', fmt(rec.n)], ['P (state)', fmt(rec.p)], ['K (state)', fmt(rec.k)], ['pH (state)', fmt(rec.ph)],
        ['Temperature °C', fmt(rec.temperature)], ['Rainfall mm', fmt(rec.rainfall)], ['Humidity %', fmt(rec.humidity)],
        ['Yield band*', rec.yield_band], ['Completeness*', fmt(rec.completeness)], ['Source row', rec.source_row],
      ]} />
      <p className="text-[11px] text-dim">* Derived by TerraMind from the record (per-hectare = total ÷ area; yield band = per-crop percentile). N/P/K/pH are state-level values; weather is a state-year aggregate.</p>
      {rec.flags && <div className="flex flex-wrap gap-1">{rec.flags.split(';').filter(Boolean).map((f) => <Badge key={f} tone="amber">{f.replaceAll('_', ' ')}</Badge>)}</div>}
      {item && (
        <div className="rounded-xl bg-glass-2 p-3">
          <p className="text-xs font-semibold text-ink">Why this case</p>
          <p className="mt-1 text-xs text-mute">{item.relevance_explanation}</p>
          {item.priority_explanation && <p className="mt-1 text-xs text-mute">{item.priority_explanation}</p>}
          {comps && (
            <div className="mt-2 grid gap-1.5">
              {Object.entries(comps).map(([k, c]) => (
                <div key={k} className="grid grid-cols-[130px_1fr_46px] items-center gap-2 text-[11px]">
                  <span className="text-mute">{k.replaceAll('_', ' ')}</span>
                  {c.assessed ? <ProgressBar value={c.value} /> : <span className="text-dim">not assessed (no user input)</span>}
                  <span className="text-right tabular-nums">{c.assessed ? fmt(c.value) : ''}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {text && <div><p className="text-xs font-semibold text-ink">Searchable case text (generated from this record's values)</p><p className="mt-1 rounded-xl bg-glass-2 p-3 text-xs leading-relaxed text-mute">{text}</p></div>}
    </div>
  )
}
