import { useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight, Eraser, Search } from 'lucide-react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Badge, Button, Card, Collapsible, Drawer, EmptyState, ErrorState, Field, Input, Notice, PageHeader, Select, Spinner, Table, cx } from '../components/ui'
import { CaseDetail } from '../components/CaseDetail'
import { fmt } from '../utils/format'

const RANGES = [['n', 'N'], ['p', 'P'], ['k', 'K'], ['ph', 'pH'], ['rainfall', 'Rainfall'], ['temperature', 'Temp °C'], ['humidity', 'Humidity %']]
const SORTS = [['year', 'Year'], ['yield', 'Yield'], ['area', 'Area'], ['production', 'Production'], ['rainfall', 'Rainfall'], ['ph', 'pH'], ['crop', 'Crop'], ['state', 'State']]
const EMPTY = { q: '', origin: '', crop: '', state: '', season: '', year_min: '', year_max: '', mode: 'hybrid', sort_by: 'year', order: 'desc' }

export default function CaseExplorer() {
  const opts = useApi(() => api.cases.options(), [])
  const [f, setF] = useState({ ...EMPTY, ...Object.fromEntries(RANGES.flatMap(([k]) => [[k + '_min', ''], [k + '_max', '']])) })
  const [applied, setApplied] = useState(f)
  const [page, setPage] = useState(1)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sel, setSel] = useState(null)
  const PAGE = 25
  const set = (k) => (e) => setF((x) => ({ ...x, [k]: e.target.value }))
  const searching = applied.q.trim().length > 0
  const isSearch = (d) => d?.kind === 'search'

  const filterObj = (a) => {
    const o = {}
    ;['crop', 'state', 'season'].forEach((k) => { if (a[k]) o[k] = [a[k]] })
    if (a.origin) o.origin = [a.origin]
    ;['year_min', 'year_max'].forEach((k) => { if (a[k] !== '') o[k] = Number(a[k]) })
    RANGES.forEach(([k]) => { ['_min', '_max'].forEach((s) => { if (a[k + s] !== '') o[k + s] = Number(a[k + s]) }) })
    return o
  }

  useEffect(() => {
    let alive = true
    async function load() {
      setLoading(true); setError(null)
      try {
        if (searching) {
          const r = await api.cases.search({ query: applied.q.trim(), filters: filterObj(applied), top_k: 100, mode: applied.mode })
          if (alive) setData({ kind: 'search', ...r })
        } else {
          const r = await api.cases.list({ ...filterObj(applied), sort_by: applied.sort_by, order: applied.order, page, page_size: PAGE })
          if (alive) setData({ kind: 'list', ...r })
        }
      } catch (e) { if (alive) { setError(e); setData(null) } } finally { if (alive) setLoading(false) }
    }
    load()
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applied, page])

  const apply = (e) => { e?.preventDefault(); setPage(1); setApplied({ ...f }) }
  const reset = () => { const z = { ...f, ...Object.fromEntries(Object.keys(f).map((k) => [k, EMPTY[k] ?? ''])) }; setF(z); setApplied(z); setPage(1) }

  const rows = data?.kind === 'search' ? data.items.slice((page - 1) * PAGE, page * PAGE).map((i) => ({ ...i.record, _item: i })) : data?.items?.map((r) => ({ ...r })) || []
  const total = data?.kind === 'search' ? data.items.length : data?.total || 0
  const pages = Math.max(1, Math.ceil(total / PAGE))
  const o = opts.data

  return (
    <div className="fade-up">
      <PageHeader eyebrow="Search the evidence" title="Historical Case Explorer" subtitle="Browse and search the full historical dataset. Typing a query switches to hybrid (BM25 + semantic) ranking." />
      <form onSubmit={apply}>
        <Card className="space-y-3">
          <div className="grid gap-3 md:grid-cols-[1fr_180px_auto]">
            <Field label="Search" htmlFor="s"><Input id="s" value={f.q} onChange={set('q')} placeholder="e.g. wheat Punjab rabi, acidic soil high rainfall rice" /></Field>
            <Field label="Search mode"><Select value={f.mode} onChange={set('mode')}><option value="hybrid">Hybrid (RRF)</option><option value="keyword">Keyword (BM25)</option><option value="semantic">Semantic</option></Select></Field>
            <div className="flex items-end gap-2"><Button type="submit" icon={Search}>Apply</Button><Button type="button" variant="secondary" icon={Eraser} onClick={reset}>Reset</Button></div>
          </div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Field label="Crop"><Select value={f.crop} onChange={set('crop')}><option value="">Any</option>{o?.crops.map((c) => <option key={c}>{c}</option>)}</Select></Field>
            <Field label="State"><Select value={f.state} onChange={set('state')}><option value="">Any</option>{o?.states.map((c) => <option key={c}>{c}</option>)}</Select></Field>
            <Field label="Season"><Select value={f.season} onChange={set('season')}><option value="">Any</option>{o?.seasons.map((c) => <option key={c}>{c}</option>)}</Select></Field>
            <Field label="Year from"><Input type="number" value={f.year_min} onChange={set('year_min')} min={o?.years.min} max={o?.years.max} /></Field>
            <Field label="Year to"><Input type="number" value={f.year_max} onChange={set('year_max')} min={o?.years.min} max={o?.years.max} /></Field>
          </div>
          <Collapsible title="More filters: soil and weather ranges">
<div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
            {RANGES.map(([k, l]) => (
              <Field key={k} label={`${l} (min – max)`}>
                <div className="flex gap-1"><Input type="number" step="any" placeholder={o ? String(Math.floor(o.ranges[k].min)) : ''} value={f[k + '_min']} onChange={set(k + '_min')} aria-label={`${l} min`} /><Input type="number" step="any" placeholder={o ? String(Math.ceil(o.ranges[k].max)) : ''} value={f[k + '_max']} onChange={set(k + '_max')} aria-label={`${l} max`} /></div>
              </Field>
            ))}
          </div>
          </Collapsible>
          {!searching && (
            <div className="grid grid-cols-2 gap-3 sm:w-80">
              <Field label="Sort by"><Select value={f.sort_by} onChange={(e) => { set('sort_by')(e); setApplied((a) => ({ ...a, sort_by: e.target.value })); setPage(1) }}>{SORTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</Select></Field>
              <Field label="Order"><Select value={f.order} onChange={(e) => { set('order')(e); setApplied((a) => ({ ...a, order: e.target.value })); setPage(1) }}><option value="desc">Descending</option><option value="asc">Ascending</option></Select></Field>
            </div>
          )}
        </Card>
      </form>

      <div className="mt-4 space-y-3">
        {error && <ErrorState error={error} onRetry={() => setApplied({ ...applied })} />}
        {data?.fallback && <Notice tone="amber" title="Fallback in use">{data.fallback}</Notice>}
        {data?.kind === 'search' && (
          <Notice tone="blue" title="How these are ranked">{data.scoring} {data.total_candidates_after_filter.toLocaleString()} cases matched your metadata filters; {data.filtered_out.toLocaleString()} were filtered out before ranking. Showing the top {data.items.length}.</Notice>
        )}
        <Card>
          <div className="mb-3 flex items-center justify-between text-sm">
            <span className="font-medium text-ink">{loading ? 'Loading…' : `${total.toLocaleString()} ${isSearch(data) ? 'ranked results' : 'matching records'}`}</span>
            <span className="flex items-center gap-2 text-xs text-mute">
              <Button variant="secondary" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)} aria-label="Previous page"><ChevronLeft className="h-3.5 w-3.5" /></Button>
              Page {page} / {pages}
              <Button variant="secondary" size="sm" disabled={page >= pages} onClick={() => setPage(page + 1)} aria-label="Next page"><ChevronRight className="h-3.5 w-3.5" /></Button>
            </span>
          </div>
          {loading && !rows.length ? <Spinner /> : (
            <Table
              empty="No cases match these filters"
              rows={rows} rowKey={(r) => r.case_id} onRow={setSel}
              columns={[
                ...(isSearch(data) ? [{ key: 'rank', label: '#', render: (r) => r._item.hybrid_rank }] : []),
                { key: 'case_id', label: 'Case', render: (r) => <span className="font-mono text-xs">{r.case_id}</span> },
                { key: 'crop', label: 'Crop' }, { key: 'state', label: 'State' }, { key: 'season', label: 'Season' }, { key: 'year', label: 'Year' },
                { key: 'yield', label: 'Yield', align: 'right', render: (r) => fmt(r.yield, 2) },
                { key: 'rainfall', label: 'Rain mm', align: 'right', render: (r) => fmt(r.rainfall, 0) },
                { key: 'ph', label: 'pH', align: 'right', render: (r) => fmt(r.ph, 1) },
                { key: 'band', label: 'Yield band', render: (r) => <Badge tone={r.yield_band === 'high yield' ? 'green' : r.yield_band === 'low yield' ? 'amber' : 'neutral'}>{r.yield_band}</Badge> },
                ...(isSearch(data) ? [
                  { key: 'kw', label: 'Keyword', align: 'right', render: (r) => (r._item.keyword_rank ? `#${r._item.keyword_rank} · ${fmt(r._item.keyword_score, 1)}` : '—') },
                  { key: 'sem', label: 'Semantic', align: 'right', render: (r) => (r._item.semantic_rank ? `#${r._item.semantic_rank} · ${fmt(r._item.semantic_score, 2)}` : '—') },
                  { key: 'hy', label: 'RRF score', align: 'right', render: (r) => fmt(r._item.hybrid_score, 4) },
                ] : []),
              ]}
            />
          )}
        </Card>
      </div>
      {sel && <Detail row={sel} onClose={() => setSel(null)} />}
    </div>
  )
}

function Detail({ row, onClose }) {
  const full = useApi(() => api.cases.get(row.case_id), [row.case_id])
  return (
    <Drawer open onClose={onClose} title={`Case ${row.case_id}`}>
      {full.error ? <ErrorState error={full.error} /> : full.loading ? <Spinner /> : <CaseDetail item={row._item} record={full.data} text={full.data.text} />}
    </Drawer>
  )
}
