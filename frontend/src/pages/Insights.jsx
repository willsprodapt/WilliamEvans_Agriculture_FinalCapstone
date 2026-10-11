import { useEffect, useState } from 'react'
import { BarChart3, ClipboardCheck, MessageSquareHeart, Microscope } from 'lucide-react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Badge, Button, Card, CardHeader, EmptyState, ErrorState, Field, Notice, PageHeader, Select, Skeleton, Stat, Table } from '../components/ui'
import { Donut, HBar, Lines, Scatter2, VBar } from '../components/charts'
import { dt, fmt, pct } from '../utils/format'

export default function Insights() {
  const opts = useApi(() => api.cases.options(), [])
  const [scope, setScope] = useState({ crop: 'Rice', state: '', season: '' })
  const [impact, setImpact] = useState({ data: null, loading: true, error: null })
  const fb = useApi(() => api.feedback.summary(), [])
  const adv = useApi(() => api.advisory.summary(), [])

  async function analyze(s = scope) {
    setImpact((x) => ({ ...x, loading: true, error: null }))
    try {
      const body = Object.fromEntries(Object.entries(s).filter(([, v]) => v))
      setImpact({ data: await api.predictions.impact(body), loading: false, error: null })
    } catch (e) { setImpact({ data: null, loading: false, error: e }) }
  }
  useEffect(() => { analyze() /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [])
  const a = impact.data
  const groups = Object.entries(a?.group_comparisons || {}).filter(([, v]) => v.length)
  const f = fb.data

  return (
    <div className="fade-up">
      <PageHeader eyebrow="What the data says" title="Insights & Feedback" subtitle="Statistical relationships in the historical data, what the agents found, and user feedback." />

      <Card>
        <CardHeader title="Soil, weather & farming-input relationships" subtitle="Computed on real dataset records. Correlations and importances are associations, not causal effects." icon={Microscope} />
        <div className="mb-4 grid gap-3 sm:grid-cols-4">
          <Field label="Crop"><Select value={scope.crop} onChange={(e) => setScope({ ...scope, crop: e.target.value })}><option value="">All crops (relative yield)</option>{opts.data?.crops.map((c) => <option key={c}>{c}</option>)}</Select></Field>
          <Field label="State"><Select value={scope.state} onChange={(e) => setScope({ ...scope, state: e.target.value })}><option value="">All states</option>{opts.data?.states.map((c) => <option key={c}>{c}</option>)}</Select></Field>
          <Field label="Season"><Select value={scope.season} onChange={(e) => setScope({ ...scope, season: e.target.value })}><option value="">All seasons</option>{opts.data?.seasons.map((c) => <option key={c}>{c}</option>)}</Select></Field>
          <div className="flex items-end"><Button onClick={() => analyze()} loading={impact.loading} icon={BarChart3}>Analyze</Button></div>
        </div>
        {impact.error && <ErrorState error={impact.error} />}
        {impact.loading && !a ? <Skeleton className="h-64" /> : a && !a.available ? <Notice tone="amber">{a.reason}</Notice> : a && (
          <>
            <p className="mb-3 text-xs text-mute">{a.scope.n_records.toLocaleString()} records · value analysed: {a.value_definition}</p>
            <div className="grid gap-4 lg:grid-cols-2">
              <div><p className="mb-1 text-xs font-semibold text-ink">Spearman correlation with yield <Badge tone="blue">association</Badge></p><HBar data={a.correlations.map((c) => ({ name: c.feature, value: Number(c.spearman_r.toFixed(3)) }))} signed height={270} /></div>
              <div><p className="mb-1 text-xs font-semibold text-ink">Model permutation importance (global) <Badge tone="blue">predictive</Badge></p>{a.model_feature_importance ? <HBar data={a.model_feature_importance.values.map((v) => ({ name: v.feature, value: Number(v.importance.toFixed(3)) }))} height={270} color="#059669" /> : <EmptyState title="Yield model not trained" />}</div>
            </div>
            <div className="mt-4 grid gap-4 lg:grid-cols-3">
              {groups.map(([name, rows]) => (
                <div key={name}><p className="mb-1 text-xs font-semibold text-ink">By {name.toLowerCase()}</p><VBar data={rows.map((r) => ({ name: `${r.group} (n=${r.n})`, median: Number(r.median.toFixed(3)) }))} bars={[{ key: 'median', label: 'Median' }]} height={200} /></div>
              ))}
            </div>
            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <div><p className="mb-1 text-xs font-semibold text-ink">Rainfall vs {a.scope.crop ? 'yield' : 'relative yield'} (sample of records)</p><Scatter2 data={a.scatter_rainfall} xKey="rainfall" yKey="value" xLabel="Annual rainfall (mm)" yLabel="Yield" /></div>
              {a.crop_comparison?.length > 0 && <div><p className="mb-1 text-xs font-semibold text-ink">Crops in this scope: record counts</p><HBar data={a.crop_comparison.map((c) => ({ name: c.crop, value: c.n }))} height={240} /><p className="text-[11px] text-dim">Yields are in different units per crop and are not compared across crops.</p></div>}
            </div>
            <ul className="mt-4 list-disc space-y-0.5 pl-5 text-[11px] text-dim">{a.confounding_notes.map((n) => <li key={n}>{n}</li>)}</ul>
          </>
        )}
      </Card>

      <h2 className="mb-2 mt-6 text-sm font-semibold text-ink">Agent findings & judge validation (from persisted advisory runs)</h2>
      {adv.error ? <ErrorState error={adv.error} /> : adv.loading ? <Skeleton className="h-40" /> : adv.data.n_queries === 0 ? <EmptyState title="No advisory runs yet" hint="Run a query on the Crop Advisory page and results will aggregate here." /> : (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card><CardHeader title="Judge verdicts" subtitle={`${adv.data.n_queries} queries`} /><Donut data={adv.data.judge_verdicts.map((x) => ({ name: x.verdict.replace('_', ' '), value: x.count }))} height={200} /></Card>
          <Card><CardHeader title="Rule-based risk categories" subtitle="Across all recommendations" /><Donut data={adv.data.risk_categories.map((x) => ({ name: x.category.replaceAll('_', ' '), value: x.count }))} height={200} /></Card>
          <Card><CardHeader title="Most recommended crops" /><HBar data={adv.data.top_recommended_crops.map((x) => ({ name: x.crop, value: x.count }))} height={200} /></Card>
          <Card className="lg:col-span-3"><CardHeader title="Agent outcomes" /><Table dense rows={adv.data.agent_outcomes} rowKey={(r) => r.agent + r.status} columns={[{ key: 'agent', label: 'Agent' }, { key: 'status', label: 'Status', render: (r) => <Badge tone={r.status === 'success' ? 'green' : r.status === 'failed' ? 'red' : 'neutral'}>{r.status}</Badge> }, { key: 'count', label: 'Runs', align: 'right' }]} /></Card>
        </div>
      )}

      <h2 className="mb-2 mt-6 flex items-center gap-2 text-sm font-semibold text-ink"><MessageSquareHeart className="h-4 w-4 text-accent" /> User feedback</h2>
      {fb.error ? <ErrorState error={fb.error} onRetry={fb.reload} /> : fb.loading ? <Skeleton className="h-40" /> : f.total === 0 ? <EmptyState title="No feedback yet" hint="Use the helpful / not helpful controls under each recommendation on the Crop Advisory page." /> : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat dark label="Feedback entries" value={fmt(f.total)} />
            <Stat label="Helpful rate" value={pct(f.helpful_rate)} sub={`${f.helpful} helpful · ${f.not_helpful} not helpful`} />
            <Stat label="Average rating" value={f.average_rating != null ? f.average_rating.toFixed(2) : '—'} sub="of 5, where given" />
            <Stat label="Review candidates" value={fmt(f.review_candidates.length)} sub="unhelpful, awaiting verification" />
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Card><CardHeader title="Helpful vs not helpful" /><Donut data={[{ name: 'Helpful', value: f.helpful }, { name: 'Not helpful', value: f.not_helpful }]} height={200} /></Card>
            <Card><CardHeader title="Reported problem categories" /><HBar data={f.categories.map((c) => ({ name: c.category.replaceAll('_', ' '), value: c.count }))} height={200} /></Card>
            <Card><CardHeader title="Feedback over time" /><Lines data={f.trend} x="date" lines={[{ key: 'helpful', label: 'Helpful', color: '#059669' }, { key: 'not_helpful', label: 'Not helpful', color: '#b45309' }]} height={200} /></Card>
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <Card><CardHeader title="By crop" /><VBar data={f.by_crop} x="crop" bars={[{ key: 'helpful', label: 'Helpful', color: '#059669' }, { key: 'not_helpful', label: 'Not helpful', color: '#b45309' }]} stacked height={220} /></Card>
            <Card><CardHeader title="Recent feedback" /><Table dense rows={f.recent} rowKey={(r) => r.id} columns={[{ key: 'created_at', label: 'When', render: (r) => dt(r.created_at) }, { key: 'crop', label: 'Crop', render: (r) => r.crop || '—' }, { key: 'helpful', label: '', render: (r) => <Badge tone={r.helpful ? 'green' : 'amber'}>{r.helpful ? 'helpful' : 'not helpful'}</Badge> }, { key: 'comment', label: 'Comment', render: (r) => <span className="text-xs">{r.comment}</span> }]} /></Card>
          </div>
          <Card className="mt-4">
            <CardHeader title="Review candidates → evaluation" icon={ClipboardCheck} subtitle="Unhelpful feedback to verify by a person before it becomes an evaluation case." />
            <Table dense empty="No unhelpful feedback to review" rows={f.review_candidates} rowKey={(r) => r.feedback_id} columns={[{ key: 'query_id', label: 'Query', render: (r) => <span className="font-mono text-xs">{r.query_id || '—'}</span> }, { key: 'crop', label: 'Crop', render: (r) => r.crop || '—' }, { key: 'category', label: 'Category', render: (r) => r.category?.replaceAll('_', ' ') || '—' }, { key: 'comment', label: 'Comment', render: (r) => <span className="text-xs">{r.comment}</span> }]} />
            <ol className="mt-3 list-decimal space-y-0.5 pl-5 text-[11px] text-mute">{f.improvement_workflow.map((w) => <li key={w} className="list-none -ml-5">{w}</li>)}</ol>
          </Card>
        </>
      )}
    </div>
  )
}
