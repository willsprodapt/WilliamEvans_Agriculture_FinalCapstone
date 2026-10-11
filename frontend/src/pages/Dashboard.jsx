import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowUp, Database, FlaskConical, Layers, Map, MessageSquareHeart, Search, Sparkles, Sprout, Wheat } from 'lucide-react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Badge, Card, CardHeader, ErrorState, Skeleton, Stat, Table } from '../components/ui'
import { Donut, HBar } from '../components/charts'
import { dt, fmt } from '../utils/format'

const STATUS_TONE = { completed: 'green', completed_with_warnings: 'amber', needs_clarification: 'blue', no_evidence: 'neutral', failed: 'red' }
const IDEAS = [
  ['Rabi crops for Punjab', 'Which crops perform well in Punjab during the Rabi season with soil pH 7.8 and rainfall around 600 mm?'],
  ['Acidic, high-rainfall soil', 'I farm in Kerala in the Kharif season. Soil is acidic (pH 5.7) with high rainfall. What should I consider growing?'],
  ['Alternatives to rice', 'Suggest alternatives to rice for Kharif in West Bengal with pH 6.2.'],
]

export default function Dashboard() {
  const nav = useNavigate()
  const [q, setQ] = useState('')
  const stats = useApi(() => api.system.stats(), [])
  const recent = useApi(() => api.advisory.recent(5), [])
  const s = stats.data

  function go(e, text = q) {
    e?.preventDefault()
    if (text.trim().length >= 3) nav('/advisory', { state: { query: text.trim(), autorun: true } })
  }

  return (
    <div className="fade-up">
      {/* hero */}
      <section className="relative pb-8 pt-4 sm:pt-10">
        <div className="hero-orb floaty -right-10 -top-10 h-72 w-72 bg-lime-300/20" aria-hidden />
        <div className="hero-orb -left-24 top-40 h-64 w-64 bg-emerald-400/20" aria-hidden />
        <p className="eyebrow mb-5 flex items-center gap-2"><Sparkles className="h-3.5 w-3.5" aria-hidden /> Evidence-driven crop advisory</p>
        <h1 className="font-display max-w-4xl text-5xl font-semibold leading-[1.04] tracking-tight text-ink sm:text-7xl">
          Grow the right crop,<br /><span className="text-gradient">backed by real history.</span>
        </h1>
        <p className="mt-6 max-w-2xl text-base leading-relaxed text-mute sm:text-lg">
          Ask in plain language. TerraMind searches {s ? fmt(s.records) : 'tens of thousands of'} historical farming cases, ranks the crops that fit your
          conditions, and shows the exact evidence, risk and yield estimate behind every suggestion.
        </p>

        <form onSubmit={go} className="glass ring-glow relative mt-10 max-w-4xl rounded-[28px] p-3 sm:p-4" aria-label="Ask TerraMind">
          <div className="flex items-end gap-3">
            <textarea value={q} onChange={(e) => setQ(e.target.value)} rows={2} maxLength={1000} aria-label="Describe your farm conditions"
              placeholder="Describe your farm: state, season, soil pH, rainfall…  e.g. “Which crops suit Punjab in Rabi?”"
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) go(e) }}
              className="w-full resize-none bg-transparent px-3 py-2 text-base text-ink placeholder:text-dim focus:outline-none" />
            <button type="submit" disabled={q.trim().length < 3} aria-label="Get advice"
              className="flex h-12 shrink-0 items-center gap-2 rounded-full bg-gradient-to-b from-[#d6f984] to-[#a9e046] px-5 text-sm font-semibold text-deep shadow-[0_10px_28px_-8px_rgba(196,242,106,0.7)] transition hover:brightness-105 active:scale-[.98] disabled:bg-white/15 disabled:from-white/15 disabled:to-white/15 disabled:text-white/40 disabled:shadow-none">
              Get advice <ArrowUp className="h-4 w-4 rotate-45" />
            </button>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-3">
            <span className="text-[11px] uppercase tracking-wider text-dim">Try</span>
            {IDEAS.map(([label, text]) => (
              <button type="button" key={label} onClick={() => setQ(text)} className="rounded-full border border-line bg-white/[0.03] px-3.5 py-1.5 text-xs text-mute transition hover:border-accent/50 hover:bg-accent/10 hover:text-ink">{label}</button>
            ))}
          </div>
        </form>
      </section>

      {/* feature cards */}
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label="Sections">
        {[
          { to: '/advisory', icon: Sprout, t: 'Crop Advisory', d: 'Ranked crops with the historical cases, risk and yield estimate behind each one.' },
          { to: '/cases', icon: Search, t: 'Case Explorer', d: `Search and filter ${s ? fmt(s.records) : ''} cases with hybrid keyword + semantic ranking.` },
          { to: '/evaluation', icon: FlaskConical, t: 'Evaluation', d: 'Measured retrieval, model and DeepEval results. Nothing estimated.' },
          { to: '/insights', icon: MessageSquareHeart, t: 'Insights & Feedback', d: 'Soil, weather and input relationships, plus what users told us.' },
        ].map(({ to, icon: I, t, d }) => (
          <Link key={to} to={to} className="glass lift group rounded-3xl p-6">
            <span className="mb-5 flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-accent/25 to-emerald-400/10 text-accent"><I className="h-5 w-5" aria-hidden /></span>
            <p className="font-display flex items-center justify-between text-base font-semibold text-ink">{t}<ArrowUp className="h-4 w-4 rotate-45 text-dim transition group-hover:text-accent" aria-hidden /></p>
            <p className="mt-2 text-[13px] leading-relaxed text-mute">{d}</p>
          </Link>
        ))}
      </section>

      <ErrorState error={stats.error} onRetry={stats.reload} />

      {/* numbers */}
      <section className="mt-10">
        <p className="eyebrow mb-2">By the numbers</p>
        <h2 className="font-display mb-5 text-3xl font-semibold text-ink">The data behind the advice</h2>
        {stats.loading ? <div className="grid grid-cols-2 gap-3 lg:grid-cols-4"><Skeleton /><Skeleton /><Skeleton /><Skeleton /></div> : s && (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Stat dark label="Farming cases" value={fmt(s.records)} icon={Database} sub="Searchable by meaning and keywords" />
            <Stat label="Crop types" value={fmt(s.crops)} icon={Wheat} sub={`across ${s.seasons} seasons`} />
            <Stat label="States / regions" value={fmt(s.states)} icon={Map} sub={`${s.years[0]}–${s.years[1]}`} />
            <Stat label="Questions answered" value={fmt(s.advisory_queries_processed)} icon={Layers} sub={`${fmt(s.feedback_count)} feedback entries`} />
          </div>
        )}
        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader title="Records per crop" subtitle="Top 10 crops by number of historical cases" />
            {stats.loading ? <Skeleton className="h-64" /> : <HBar data={(s?.crop_counts || []).slice(0, 10).map((c) => ({ name: c.name, value: c.count }))} height={290} />}
          </Card>
          <Card>
            <CardHeader title="Cases by season" subtitle="How the history is spread across growing seasons" />
            {stats.loading ? <Skeleton className="h-64" /> : <Donut data={(s?.season_counts || []).map((c) => ({ name: c.name, value: c.count }))} height={290} />}
          </Card>
        </div>
      </section>

      <section className="mt-6">
        <Card>
          <CardHeader title="Recent questions" action={<Link to="/advisory" className="text-xs font-medium text-accent hover:underline">Ask a new one</Link>} />
          {recent.loading ? <Skeleton className="h-32" /> : recent.error ? <ErrorState error={recent.error} onRetry={recent.reload} /> : (
            <Table dense empty="No questions yet — ask one above" rows={recent.data?.items || []} rowKey={(r) => r.id}
              columns={[
                { key: 'query', label: 'Question', render: (r) => <span className="line-clamp-1 max-w-xl">{r.query}</span> },
                { key: 'status', label: 'Status', render: (r) => <Badge tone={STATUS_TONE[r.status] || 'neutral'}>{r.status.replaceAll('_', ' ')}</Badge> },
                { key: 'created_at', label: 'When', render: (r) => dt(r.created_at) },
                { key: 'duration_ms', label: 'Time', align: 'right', render: (r) => `${(r.duration_ms / 1000).toFixed(1)}s` },
              ]} />
          )}
        </Card>
      </section>
    </div>
  )
}
