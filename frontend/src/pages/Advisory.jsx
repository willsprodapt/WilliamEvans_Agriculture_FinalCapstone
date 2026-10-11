import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { AlertTriangle, Bot, CheckCircle2, CircleDashed, Eraser, FlaskConical, Loader2, Search, Sparkles, XCircle } from 'lucide-react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Badge, Button, Card, CardHeader, Collapsible, Drawer, EmptyState, ErrorState, Field, Input, KV, Notice, PageHeader, ProgressBar, Select, Table, Textarea, Tooltip, cx } from '../components/ui'
import { HBar, VBar } from '../components/charts'
import { CaseDetail } from '../components/CaseDetail'
import FeedbackBox from '../components/FeedbackBox'
import { FIELD_LABELS, PROVENANCE, RISK_TONE, VERDICT_LABEL, VERDICT_TONE, fmt } from '../utils/format'

const EXAMPLES = [
  'Which crops perform well in Punjab during the Rabi season with soil pH 7.8, rainfall around 600 mm and nitrogen 150?',
  'I farm in Kerala in the Kharif season. Soil is acidic (pH 5.7) with high rainfall. What should I consider growing?',
  'Suggest alternatives to rice for Kharif in West Bengal with pH 6.2.',
]
const NUM_FIELDS = [
  ['n', 'Nitrogen N', 0, 500], ['p', 'Phosphorus P', 0, 300], ['k', 'Potassium K', 0, 300], ['ph', 'Soil pH', 3, 10],
  ['rainfall', 'Rainfall (mm/yr)', 0, 12000], ['temperature', 'Temp (°C)', -10, 50], ['humidity', 'Humidity (%)', 0, 100],
]
const EMPTY = { query: '', crop: '', state: '', season: '', n: '', p: '', k: '', ph: '', rainfall: '', temperature: '', humidity: '', fertilizer: '', pesticide: '', area: '', top_k: 15, use_llm: true }

function StageIcon({ s }) {
  if (s === 'success') return <CheckCircle2 className="h-4 w-4 text-accent" aria-label="done" />
  if (s === 'running') return <Loader2 className="h-4 w-4 animate-spin text-accent" aria-label="running" />
  if (s === 'failed') return <XCircle className="h-4 w-4 text-red-300" aria-label="failed" />
  if (s === 'skipped') return <CircleDashed className="h-4 w-4 text-dim" aria-label="skipped" />
  return <CircleDashed className="h-4 w-4 text-white/20" aria-label="pending" />
}

function Progress({ stages }) {
  return (
    <Card>
      <CardHeader title="Working on it" subtitle="Each step turns green only after it has actually completed on the server" icon={Bot} />
      <ol className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
        {stages.map((s, i) => (
          <li key={s.key} className="flex items-start gap-3 text-sm">
            <StageIcon s={s.status} />
            <div className="min-w-0 flex-1">
              <p className={cx('font-medium', s.status === 'pending' ? 'text-dim' : 'text-ink')}>{i + 1}. {s.label}</p>
              {s.message && <p className="truncate text-[11px] text-mute">{s.message}</p>}
            </div>
            {s.duration_ms != null && <span className="text-[11px] tabular-nums text-dim">{(s.duration_ms / 1000).toFixed(1)}s</span>}
          </li>
        ))}
      </ol>
    </Card>
  )
}

function Conditions({ c }) {
  const found = Object.keys(FIELD_LABELS).map((k) => ({ key: k, ...c[k] })).filter((r) => r.provenance !== 'missing')
  return (
    <Card>
      <CardHeader title="What I understood" subtitle="Detected conditions and where each value came from" />
      <div className="flex flex-wrap gap-2">
        {found.length ? found.map((r) => (
          <span key={r.key} className="glass inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-xs">
            <span className="text-mute">{FIELD_LABELS[r.key]}</span><b className="text-ink">{String(r.value)}</b>
            <Badge tone={PROVENANCE[r.provenance].tone}>{PROVENANCE[r.provenance].label}</Badge>
          </span>
        )) : <span className="text-sm text-mute">No explicit conditions found.</span>}
        {c.excluded_crops.map((x) => <Badge key={x} tone="red">exclude: {x}</Badge>)}
        {[...c.objectives, ...c.constraints].map((x) => <Badge key={x} tone="blue">{x}</Badge>)}
      </div>
      {c.missing_information.length > 0 && <p className="mt-4 text-xs text-mute"><span className="text-dim">Not provided: </span>{c.missing_information.join(', ')}</p>}
    </Card>
  )
}

function RecCard({ r, queryId, openCase }) {
  const risk = r.risk, yp = r.yield_prediction, hy = r.historical_yield
  return (
    <Card className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-4">
          <span className="font-display flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-[#d6f984] to-emerald-400 text-xl font-semibold text-deep shadow-[0_8px_22px_-8px_rgba(196,242,106,.7)]">{r.rank}</span>
          <div>
            <h3 className="font-display text-2xl font-semibold text-ink">{r.crop}</h3>
            <p className="text-[11px] text-mute">{r.user_requested ? 'The crop you asked about · ' : ''}{r.n_supporting_cases} supporting cases{r.n_matching_records > r.n_supporting_cases ? ` · ${r.n_matching_records.toLocaleString()} matching records in history` : ''}</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge tone={RISK_TONE[risk?.category] || 'neutral'} title={risk?.description}>Risk: {(risk?.category || 'n/a').replaceAll('_', ' ')}</Badge>
          <Badge tone={VERDICT_TONE[r.judge?.verdict] || 'neutral'} title={r.judge?.notes}>Judge: {VERDICT_LABEL[r.judge?.verdict] || 'Not run'}</Badge>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-2xl border border-line bg-white/[0.04] p-4">
          <p className="flex items-center gap-1 text-[11px] font-medium text-mute">Evidence score <Tooltip text={r.score_method} /></p>
          <p className="font-display mt-1 text-2xl font-semibold tabular-nums text-ink">{fmt(r.score, 3)}</p>
          <ProgressBar value={r.score} />
          <p className="mt-1.5 text-[10px] text-dim">Heuristic ranking, not a probability</p>
        </div>
        <div className="rounded-2xl border border-line bg-white/[0.04] p-4">
          <p className="text-[11px] font-medium text-mute">Historical yield · actual</p>
          <p className="font-display mt-1 text-2xl font-semibold tabular-nums text-ink">{fmt(hy.median, 2)}</p>
          <p className="mt-1.5 text-[10px] text-dim">median of {hy.n} cases · range {fmt(hy.min, 2)}–{fmt(hy.max, 2)} · source units</p>
        </div>
        <div className="rounded-2xl border border-dashed border-accent/35 bg-accent/[0.05] p-4">
          <p className="text-[11px] font-medium text-accent">Predicted yield · model estimate</p>
          {yp?.available ? (
            <>
              <p className="font-display mt-1 text-2xl font-semibold tabular-nums text-ink">{fmt(yp.predicted_yield, 2)}</p>
              <p className="mt-1.5 text-[10px] text-dim">10–90% range {fmt(yp.interval_10_90.low, 2)}–{fmt(yp.interval_10_90.high, 2)} · estimate, not observed</p>
            </>
          ) : <p className="mt-2 text-xs text-mute">Unavailable: {yp?.reason || 'not computed'}</p>}
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-mute">Why this crop</p>
        <ul className="space-y-1.5 text-sm text-ink/90">{r.reasons.map((x, i) => <li key={i} className="flex gap-2"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-accent" />{x}</li>)}</ul>
      </div>

      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-mute">Supporting cases</p>
        <div className="flex flex-wrap gap-1.5">
          {r.supporting_case_ids.map((id) => (
            <button key={id} onClick={() => openCase(id)} className="rounded-full border border-line px-3 py-1 font-mono text-[11px] text-mute transition hover:border-accent/40 hover:text-ink">{id}</button>
          ))}
        </div>
      </div>

      {risk && (
        <Collapsible title="Suitability & risk details" right={<span className="text-xs">{risk.score != null ? `score ${fmt(risk.score, 2)}` : 'insufficient information'}</span>}>
          <Notice tone="blue" title="Method">{risk.description} <b>Rule-based heuristic — not a trained classifier.</b></Notice>
          <div className="mt-3"><Table dense rows={risk.factors || []} rowKey={(f) => f.factor} columns={[
            { key: 'factor', label: 'Factor' },
            { key: 'user_value', label: 'Your value', render: (f) => (f.user_value == null ? '—' : String(f.user_value)) },
            { key: 'range', label: 'Typical (P10–P90)', render: (f) => (f.historical_p10 != null ? `${f.historical_p10}–${f.historical_p90}` : '—') },
            { key: 'status', label: 'Assessment', render: (f) => <span className={cx('text-xs', f.score === 1 ? 'text-accent' : f.score === 0 ? 'text-red-300' : 'text-amber-200')}>{f.status}</span> },
          ]} /></div>
          {risk.missing_conditions?.length > 0 && <p className="mt-2 text-xs text-mute">Not assessed (missing): {risk.missing_conditions.join(', ')}</p>}
        </Collapsible>
      )}
      {yp?.available && yp.assumptions?.length > 0 && (
        <Collapsible title={`Yield prediction assumptions (${yp.assumptions.length})`}>
          <ul className="list-disc space-y-0.5 pl-5 text-xs text-mute">{yp.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
          <p className="mt-2 text-[11px] text-dim">{yp.model_name} · {yp.feature_set} features · {yp.interval_10_90.method}</p>
        </Collapsible>
      )}
      <div className="border-t border-line pt-4"><FeedbackBox compact queryId={queryId} crop={r.crop} recommendationId={r.recommendation_id} caseIds={r.supporting_case_ids} /></div>
    </Card>
  )
}

function Impact({ a }) {
  if (!a) return null
  if (!a.available) return <Card><CardHeader title="Soil, weather & input impact" /><Notice tone="amber">{a.reason}</Notice></Card>
  const groups = Object.entries(a.group_comparisons || {}).filter(([, v]) => v.length)
  return (
    <Card>
      <CardHeader title="Soil, weather & input impact" subtitle={`${[a.scope.crop, a.scope.state, a.scope.season].filter(Boolean).join(' · ') || 'All'} · ${a.scope.n_records} records · ${a.value_definition}`} icon={FlaskConical} />
      {a.note && <div className="mb-3"><Notice tone="blue">{a.note}</Notice></div>}
      <div className="grid gap-6 lg:grid-cols-2">
        <div><p className="mb-2 text-xs font-semibold text-ink">Rank correlation with yield <Badge tone="blue">association, not causation</Badge></p>
          <HBar data={a.correlations.map((c) => ({ name: c.feature, value: Number(c.spearman_r.toFixed(3)) }))} signed height={250} /></div>
        <div><p className="mb-2 text-xs font-semibold text-ink">Model feature importance <Badge tone="blue">predictive, not causal</Badge></p>
          {a.model_feature_importance ? <HBar data={a.model_feature_importance.values.slice(0, 8).map((v) => ({ name: v.feature, value: Number(v.importance.toFixed(3)) }))} height={250} color="#34d399" /> : <EmptyState title="Model not trained" />}</div>
      </div>
      <div className="mt-4"><Collapsible title="Group comparisons & caveats">
        <div className="grid gap-4 lg:grid-cols-2">
          {groups.slice(0, 4).map(([name, rows]) => (
            <div key={name}><p className="mb-1 text-xs font-semibold text-ink">Median {a.scope.crop ? 'yield' : 'relative yield'} by {name.toLowerCase()}</p>
              <VBar data={rows.map((r) => ({ name: `${r.group} (n=${r.n})`, median: Number(r.median.toFixed(3)) }))} bars={[{ key: 'median', label: 'Median' }]} height={190} /></div>
          ))}
        </div>
        <ul className="mt-4 list-disc space-y-0.5 pl-5 text-[11px] text-dim">{a.confounding_notes.map((n) => <li key={n}>{n}</li>)}</ul>
      </Collapsible></div>
    </Card>
  )
}

export default function Advisory() {
  const loc = useLocation()
  const opts = useApi(() => api.cases.options(), [])
  const [form, setForm] = useState({ ...EMPTY, query: loc.state?.query || '' })
  const [errors, setErrors] = useState({})
  const [stages, setStages] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [caseView, setCaseView] = useState(null)
  const timer = useRef(null)
  const auto = useRef(!!loc.state?.autorun)
  useEffect(() => () => clearTimeout(timer.current), [])
  useEffect(() => { if (auto.current && form.query) { auto.current = false; submit() } /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [])
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === 'checkbox' ? e.target.checked : e.target.value }))

  function validate() {
    const er = {}
    if (form.query.trim().length < 3) er.query = 'Please enter a question (at least 3 characters).'
    NUM_FIELDS.forEach(([k, label, lo, hi]) => {
      if (form[k] !== '') { const v = Number(form[k]); if (Number.isNaN(v) || v < lo || v > hi) er[k] = `${label}: ${lo}–${hi}` }
    })
    ;['fertilizer', 'pesticide'].forEach((k) => { if (form[k] !== '' && !(Number(form[k]) >= 0)) er[k] = 'Must be ≥ 0' })
    if (form.area !== '' && !(Number(form.area) > 0)) er.area = 'Area must be > 0'
    setErrors(er)
    return Object.keys(er).length === 0
  }
  function payload() {
    const p = { query: form.query.trim(), top_k: Number(form.top_k), use_llm: form.use_llm }
    ;['crop', 'state', 'season'].forEach((k) => { if (form[k]) p[k] = form[k] })
    ;[...NUM_FIELDS.map((f) => f[0]), 'fertilizer', 'pesticide', 'area'].forEach((k) => { if (form[k] !== '') p[k] = Number(form[k]) })
    return p
  }
  async function poll(jobId) {
    try {
      const j = await api.advisory.job(jobId)
      setStages(j.stages)
      if (j.status === 'running') { timer.current = setTimeout(() => poll(jobId), 700); return }
      if (j.status === 'failed') setError(new Error(j.error || 'The advisory job failed')); else setResult(j.result)
      setBusy(false)
    } catch (e) { setError(e); setBusy(false) }
  }
  async function submit(e) {
    e?.preventDefault()
    if (!validate()) return
    setBusy(true); setError(null); setResult(null); setStages(null)
    try { const j = await api.advisory.startJob(payload()); setStages([]); poll(j.job_id) } catch (err) { setError(err); setBusy(false) }
  }
  function clear() { clearTimeout(timer.current); setForm(EMPTY); setErrors({}); setResult(null); setStages(null); setError(null); setBusy(false) }
  const o = opts.data

  return (
    <div className="fade-up">
      <PageHeader eyebrow="Ask TerraMind" title="Crop Advisory" subtitle="Describe your farm in plain language. TerraMind finds comparable historical cases and explains what the evidence supports." />

      <form onSubmit={submit} aria-label="Advisory query form">
        <Card className="space-y-5">
          <Field label="Your question" htmlFor="q" hint={`${form.query.length}/1000`}>
            <Textarea id="q" rows={3} value={form.query} onChange={set('query')} placeholder="e.g. Which crops perform well in Punjab during Rabi with soil pH 7.8 and about 600 mm rainfall?" aria-invalid={!!errors.query} className="text-[15px]" />
            {errors.query && <p role="alert" className="mt-1 text-xs text-red-300">{errors.query}</p>}
          </Field>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => <button type="button" key={ex} onClick={() => setForm((f) => ({ ...f, query: ex }))} className="max-w-xs truncate rounded-full border border-line px-3.5 py-1.5 text-xs text-mute transition hover:border-accent/40 hover:text-ink" title={ex}>{ex}</button>)}
          </div>
          <Collapsible title="Refine with filters & measured conditions (optional)">
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label="Crop"><Select value={form.crop} onChange={set('crop')}><option value="">Any crop</option>{o?.crops.map((c) => <option key={c}>{c}</option>)}</Select></Field>
              <Field label="State"><Select value={form.state} onChange={set('state')}><option value="">Any state</option>{o?.states.map((c) => <option key={c}>{c}</option>)}</Select></Field>
              <Field label="Season"><Select value={form.season} onChange={set('season')}><option value="">Any season</option>{o?.seasons.map((c) => <option key={c}>{c}</option>)}</Select></Field>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
              {NUM_FIELDS.map(([k, label, lo, hi]) => (
                <Field key={k} label={label} htmlFor={k}>
                  <Input id={k} type="number" step="any" min={lo} max={hi} value={form[k]} onChange={set(k)} aria-invalid={!!errors[k]} />
                  {errors[k] && <p role="alert" className="mt-0.5 text-[11px] text-red-300">{errors[k]}</p>}
                </Field>
              ))}
            </div>
            <div className="mt-4 grid grid-cols-3 gap-3 sm:w-2/3">
              <Field label="Area (ha)" htmlFor="area"><Input id="area" type="number" step="any" min="0" value={form.area} onChange={set('area')} /></Field>
              <Field label="Fertilizer (total)" htmlFor="fert"><Input id="fert" type="number" step="any" min="0" value={form.fertilizer} onChange={set('fertilizer')} /></Field>
              <Field label="Pesticide (total)" htmlFor="pest"><Input id="pest" type="number" step="any" min="0" value={form.pesticide} onChange={set('pesticide')} /></Field>
            </div>
            <p className="mt-2 text-[11px] text-dim">Area + fertilizer + pesticide together switch the yield model to its “full” feature set.</p>
            {['area', 'fertilizer', 'pesticide'].map((k) => errors[k] && <p key={k} role="alert" className="text-[11px] text-red-300">{errors[k]}</p>)}
          </Collapsible>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-5 text-xs text-mute">
              <label className="flex items-center gap-2"><input type="checkbox" checked={form.use_llm} onChange={set('use_llm')} className="accent-lime-300" /> Use Claude for explanations & judge</label>
            </div>
            <div className="flex gap-2">
              <Button type="button" variant="secondary" icon={Eraser} onClick={clear}>Clear</Button>
              <Button type="submit" icon={Search} loading={busy}>Get advice</Button>
            </div>
          </div>
        </Card>
      </form>

      <div className="mt-6 space-y-5">
        {error && <ErrorState error={error} title="Advisory request failed" />}
        {stages && busy && <Progress stages={stages} />}
        {!stages && !result && !error && <EmptyState icon={Sparkles} title="Ask a question to begin" hint="You will get ranked crops with the historical cases behind each one, risk, a yield estimate, impact analysis and validation." />}

        {result?.status === 'needs_clarification' && <Notice tone="amber" title="More information needed">{result.clarification}</Notice>}
        {result?.status === 'no_evidence' && <Notice tone="amber" title="No evidence found">{result.clarification}</Notice>}
        {result?.warnings?.length > 0 && (
          <Notice tone="amber" title="Heads up">
            <ul className="mt-1 list-disc space-y-0.5 pl-4">{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
          </Notice>
        )}
        {result?.conditions && <Conditions c={result.conditions} />}

        {result?.narrative && (
          <Card>
            <CardHeader title="Summary" subtitle={result.narrative.source === 'llm' ? `Written by Claude (${result.narrative.model}) from the evidence below` : 'Template summary (Claude not used)'} icon={Sparkles} />
            <p className="text-[15px] leading-relaxed text-ink">{result.narrative.summary}</p>
            {result.narrative.caveats?.length > 0 && <ul className="mt-3 list-disc pl-5 text-xs text-mute">{result.narrative.caveats.map((c) => <li key={c}>{c}</li>)}</ul>}
          </Card>
        )}

        {result?.recommendations?.length > 0 && (
          <div className="space-y-4">
            <h2 className="font-display text-xl font-semibold text-ink">Recommended crops</h2>
            {result.recommendations.map((r) => (
              <div key={r.crop}>
                <RecCard r={r} queryId={result.query_id} openCase={(id) => setCaseView(result.cases.find((c) => c.case_id === id) || { case_id: id, record: null })} />
                {result.narrative?.per_crop?.find((p) => p.crop === r.crop) && <p className="mx-2 mt-3 text-sm text-mute">{result.narrative.per_crop.find((p) => p.crop === r.crop).explanation}</p>}
              </div>
            ))}
          </div>
        )}

        {result?.cases?.length > 0 && (
          <Card>
            <CardHeader title="Similar historical cases" subtitle={`${result.cases.length} shown of ${result.retrieval.total_candidates_after_filter.toLocaleString()} candidates after filters · click a row for the full record`} />
            <Table rows={result.cases} rowKey={(c) => c.case_id} onRow={(c) => setCaseView(c)} columns={[
              { key: 'id', label: 'Case', render: (c) => <span className="font-mono text-xs">{c.case_id}</span> },
              { key: 'crop', label: 'Crop', render: (c) => <span>{c.record.crop}</span> },
              { key: 'loc', label: 'State · season', render: (c) => `${c.record.state} · ${c.record.season}` },
              { key: 'y', label: 'Year', render: (c) => c.record.year },
              { key: 'yield', label: 'Yield', align: 'right', render: (c) => fmt(c.record.yield, 2) },
              { key: 'pr', label: 'Priority', align: 'right', render: (c) => fmt(c.priority_score, 3) },
            ]} />
            <p className="mt-3 text-[11px] text-dim">Priority is a transparent heuristic (not a probability); historical yield is never a ranking input.</p>
          </Card>
        )}

        {result?.impact_analysis && <Impact a={result.impact_analysis} />}

        {result?.judge && (
          <Card>
            <CardHeader title="Validation" subtitle="Deterministic checks always run; the Claude judge is an extra check, not ground truth." icon={CheckCircle2} />
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Badge tone={VERDICT_TONE[result.judge.verdict]}>Overall: {VERDICT_LABEL[result.judge.verdict]}</Badge>
              <Badge tone={result.judge.deterministic?.passed ? 'green' : 'red'}>Deterministic checks: {result.judge.deterministic?.passed ? 'passed' : 'issues found'}</Badge>
              <Badge tone={result.judge.llm_judge ? 'green' : 'neutral'}>LLM judge: {String(result.judge.llm_judge_status).slice(0, 48)}</Badge>
            </div>
            <Collapsible title="Judge details & checks performed">
              {result.judge.llm_judge && (
                <div className="mb-3 space-y-2 text-sm">
                  <KV items={[['Evidence support', result.judge.llm_judge.evidence_support], ['Condition match', result.judge.llm_judge.condition_match], ['Risk disclosure', result.judge.llm_judge.risk_disclosure], ['Historical vs predicted separated', result.judge.llm_judge.historical_vs_predicted_distinguished ? 'yes' : 'no']]} />
                  <p className="text-ink/90">{result.judge.llm_judge.explanation}</p>
                  {['constraint_violations', 'unsupported_claims', 'missing_information'].map((k) => result.judge.llm_judge[k]?.length > 0 && (
                    <div key={k}><p className="text-xs font-semibold text-ink">{k.replaceAll('_', ' ')}</p><ul className="list-disc pl-5 text-xs text-mute">{result.judge.llm_judge[k].map((x) => <li key={x}>{x}</li>)}</ul></div>
                  ))}
                </div>
              )}
              {[...(result.judge.deterministic?.constraint_violations || []), ...(result.judge.deterministic?.unsupported_claims || [])].map((x) => <p key={x} className="text-xs text-red-300">• {x}</p>)}
              <ul className="mt-2 list-disc pl-5 text-[11px] text-dim">{result.judge.deterministic?.checks_run.map((c) => <li key={c}>{c}</li>)}</ul>
            </Collapsible>
          </Card>
        )}

        {result?.agents?.length > 0 && (
          <Collapsible title={`Agent status · query ${result.query_id} · ${(result.metadata.duration_ms / 1000).toFixed(1)}s · dataset ${result.metadata.dataset_version}`}>
            <Table dense rows={result.agents} rowKey={(a) => a.name} columns={[
              { key: 'name', label: 'Agent' },
              { key: 'status', label: 'Status', render: (a) => <Badge tone={a.status === 'success' ? 'green' : a.status === 'failed' ? 'red' : 'neutral'}>{a.status}</Badge> },
              { key: 'duration_ms', label: 'Time', align: 'right', render: (a) => (a.duration_ms != null ? `${(a.duration_ms / 1000).toFixed(1)}s` : '—') },
              { key: 'detail', label: 'Detail', render: (a) => <span className="text-xs text-mute">{String(a.detail || '').slice(0, 120)}</span> },
            ]} />
          </Collapsible>
        )}

        {result && result.status !== 'failed' && (
          <Card>
            <CardHeader title="Was this useful?" subtitle="Feedback is stored for human review; it does not retrain the system automatically." />
            <FeedbackBox queryId={result.query_id} caseIds={result.cases.slice(0, 10).map((c) => c.case_id)} />
          </Card>
        )}
      </div>

      {caseView && <CaseDrawer c={caseView} onClose={() => setCaseView(null)} />}
    </div>
  )
}

function CaseDrawer({ c, onClose }) {
  const full = useApi(() => api.cases.get(c.case_id), [c.case_id])
  return (
    <Drawer open onClose={onClose} title={`Case ${c.case_id}`}>
      {full.error ? <ErrorState error={full.error} /> : full.loading ? <p className="text-sm text-mute">Loading…</p> : <CaseDetail item={c.record ? c : null} record={full.data} text={full.data?.text} />}
    </Drawer>
  )
}
