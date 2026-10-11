import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, FlaskConical, Play, RefreshCw, XCircle } from 'lucide-react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Badge, Button, Card, CardHeader, Drawer, EmptyState, ErrorState, Field, KV, Notice, PageHeader, ProgressBar, Select, Skeleton, Spinner, Table, Tooltip } from '../components/ui'
import { NOT_EVALUATED, dt, fmt, metric } from '../utils/format'

const EXPLAIN = {
  faithfulness: 'DeepEval: share of claims in the generated answer that are supported by the retrieved case texts. Low = hallucination.',
  answer_relevancy: 'DeepEval: how well the answer addresses the question asked.',
  contextual_precision: 'DeepEval: whether the most relevant retrieved cases are ranked above irrelevant ones.',
  contextual_recall: 'DeepEval: whether the retrieved cases contain what is needed to produce the reference answer.',
  precision: 'Share of the top-K retrieved cases that are truly relevant (verified from structured dataset filters).',
  recall: 'Share of ALL relevant cases found in the top-K. Capped by K when more than K relevant cases exist.',
  mrr: 'Mean reciprocal rank of the first relevant case (1 = first result is relevant).',
  hit: 'Share of queries with at least one relevant case in the top-K.',
  mae: 'Mean absolute error of the yield model on held-out test years, in source yield units (dominated by high-magnitude crops).',
  rmse: 'Root mean squared error on held-out test years (source units); penalises large misses.',
  r2: 'Share of yield variance explained on held-out test years (original scale; see log scale for scale-free view).',
}

function M({ label, value, tip, sub, tone }) {
  const na = value === NOT_EVALUATED
  return (
    <Card className="!p-4">
      <p className="flex items-center gap-1 text-[11px] font-medium text-mute">{label}<Tooltip text={tip} /></p>
      <p className={`font-display mt-2 text-2xl font-semibold tabular-nums ${na ? 'text-dim' : tone === 'bad' ? 'text-red-300' : 'text-gradient'}`}>{value}</p>
      {sub && <p className="mt-0.5 text-[10px] leading-snug text-dim">{sub}</p>}
    </Card>
  )
}

export default function Evaluation() {
  const overview = useApi(() => api.evaluation.overview(), [])
  const runs = useApi(() => api.evaluation.runs(), [])
  const [runId, setRunId] = useState('')
  const [cmpId, setCmpId] = useState('')
  const [limit, setLimit] = useState(10)
  const [job, setJob] = useState(null)
  const [startErr, setStartErr] = useState(null)
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])

  useEffect(() => { if (!runId && runs.data?.items?.length) setRunId(runs.data.items[0].run_id) }, [runs.data, runId])
  const run = useApi(() => (runId ? api.evaluation.run(runId) : Promise.resolve(null)), [runId])
  const cmp = useApi(() => (cmpId ? api.evaluation.run(cmpId) : Promise.resolve(null)), [cmpId])

  function refresh() { overview.reload(); runs.reload(); run.reload() }

  async function start() {
    setStartErr(null)
    try {
      const r = await api.evaluation.start({ deepeval_limit: limit })
      setJob({ run_id: r.run_id, status: 'running', done: 0, total: 0, message: 'starting' })
      poll(r.run_id)
    } catch (e) { setStartErr(e) }
  }
  async function poll(id) {
    try {
      const j = await api.evaluation.run(id)
      if (j.status === 'running') { setJob({ run_id: id, ...j }); timer.current = setTimeout(() => poll(id), 2000); return }
      setJob(j.status === 'failed' ? { run_id: id, ...j } : null)
      await runs.reload(); setRunId(id); overview.reload()
    } catch (e) { setStartErr(e); setJob(null) }
  }

  const r = run.data
  const s = r?.summary
  const ym = overview.data?.yield_model
  const yTest = ym?.comparison?.full?.selected?.test
  const yCore = ym?.comparison?.core?.selected?.test

  return (
    <div className="fade-up">
      <PageHeader eyebrow="Measured, not estimated" title="Evaluation & Performance" subtitle="Measured results only. Anything that has not been run shows “Not evaluated”."
        actions={<>
          <Button variant="secondary" icon={RefreshCw} onClick={refresh}>Refresh</Button>
          <Select value={limit} onChange={(e) => setLimit(Number(e.target.value))} aria-label="DeepEval cases" className="w-52">{[0, 5, 10, 20].map((n) => <option key={n} value={n}>DeepEval on {n} cases</option>)}</Select>
          <Button icon={Play} onClick={start} disabled={!!job && job.status === 'running'}>Run evaluation</Button>
        </>} />
      <ErrorState error={startErr || overview.error} />
      {job && (
        <Card className="mb-4">
          <p className="mb-2 text-sm font-medium text-ink">{job.status === 'failed' ? `Evaluation failed: ${job.error}` : `Running ${job.run_id}: ${job.done}/${job.total || '…'} — ${job.message}`}</p>
          {job.status !== 'failed' && <ProgressBar value={job.total ? job.done / job.total : 0.02} />}
          <p className="mt-1 text-[11px] text-dim">Each case runs the real pipeline (and Claude when configured); this takes a few minutes.</p>
        </Card>
      )}

      {overview.data?.status === 'not_evaluated' && !job && (
        <div className="mb-4"><Notice tone="amber" title="Not evaluated">No evaluation run exists yet. Click “Run evaluation” or run <code>python scripts/run_evaluation.py</code>. The yield-model metrics below come from the training script and are real held-out results.</Notice></div>
      )}

      <Card>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Evaluation run"><Select value={runId} onChange={(e) => setRunId(e.target.value)} disabled={!runs.data?.items?.length}>{!runs.data?.items?.length && <option>No runs</option>}{runs.data?.items?.map((x) => <option key={x.run_id} value={x.run_id}>{x.run_id} · {dt(x.created_at)}</option>)}</Select></Field>
          <Field label="Compare with"><Select value={cmpId} onChange={(e) => setCmpId(e.target.value)} disabled={(runs.data?.items?.length || 0) < 2}><option value="">— none —</option>{runs.data?.items?.filter((x) => x.run_id !== runId).map((x) => <option key={x.run_id} value={x.run_id}>{x.run_id}</option>)}</Select></Field>
        </div>
        <div className="mt-5 border-t border-line pt-5">
          <KV items={[['Last evaluation', r ? dt(r.created_at) : NOT_EVALUATED], ['Dataset version', r?.versions.dataset_version], ['Yield model', r?.versions.yield_model_version || ym?.meta?.model_version], ['LLM judge', r?.versions.llm_model || (r ? 'none' : null)]]} />
        </div>
      </Card>

      {run.loading ? <div className="mt-4"><Skeleton className="h-40" /></div> : run.error ? <div className="mt-4"><ErrorState error={run.error} /></div> : null}

      <h2 className="font-display mb-3 mt-9 text-lg font-semibold text-ink">DeepEval (LLM-judged, Claude)</h2>
      {s && !['faithfulness', 'answer_relevancy', 'contextual_precision', 'contextual_recall'].some((k) => s.deepeval?.[k]?.mean != null) && (
        <div className="mb-3"><Notice tone="amber" title="DeepEval scores could not be computed for this run">
          {Object.values(s.deepeval || {}).find((d) => d?.unavailable_reason)?.unavailable_reason || 'The judge model was not reachable.'} The DeepEval pipeline itself is wired and tested; once the judge model is reachable, click “Run evaluation” (DeepEval on 5 or 10 cases) to populate these scores.
        </Notice></div>
      )}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {['faithfulness', 'answer_relevancy', 'contextual_precision', 'contextual_recall'].map((k) => {
          const d = s?.deepeval?.[k]
          return <M key={k} label={k.replaceAll('_', ' ')} tip={EXPLAIN[k]} value={d?.mean != null ? metric(d.mean) : NOT_EVALUATED}
            sub={d ? (d.n_measured ? `mean of ${d.n_measured} cases · pass rate ${fmt((d.pass_rate || 0) * 100, 0)}%` : d.unavailable_reason) : undefined} />
        })}
      </div>

      <h2 className="font-display mb-3 mt-9 text-lg font-semibold text-ink">Retrieval (verified relevance labels)</h2>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <M label={`Precision@${s?.retrieval.k ?? 'K'}`} tip={EXPLAIN.precision} value={metric(s?.retrieval.precision_at_k)} sub={s ? `${s.retrieval.n_cases} constraint-retrieval queries` : undefined} />
        <M label={`Recall@${s?.retrieval.k ?? 'K'}`} tip={EXPLAIN.recall} value={metric(s?.retrieval.recall_at_k)} sub="capped by K when >K relevant cases exist" />
        <M label="MRR" tip={EXPLAIN.mrr} value={metric(s?.retrieval.mrr)} />
        <M label={`Hit rate@${s?.retrieval.k ?? 'K'}`} tip={EXPLAIN.hit} value={metric(s?.retrieval.hit_rate_at_k)} />
      </div>

      <h2 className="font-display mb-3 mt-9 text-lg font-semibold text-ink">Recommendation validation</h2>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <M label="Test cases passed" tip="Pass/fail rules differ by test type; see the test-case explorer." value={s ? `${s.n_passed} / ${s.n_test_cases}` : NOT_EVALUATED} />
        <M label="Constraint satisfaction" tip="Share of recommendation queries whose recommended crops all have historical records in the stated state+season." value={metric(s?.recommendation_validation.constraint_satisfaction_rate)} />
        <M label="Abstention / clarification" tip="Share of no-evidence queries handled by refusing to recommend / asking for clarification." value={s ? `${metric(s.recommendation_validation.abstention_pass_rate, 2)} / ${metric(s.recommendation_validation.clarification_pass_rate, 2)}` : NOT_EVALUATED} />
        <M label="Judge verdicts" tip="Claude LLM-as-judge verdict distribution across pipeline runs. A judge verdict is not ground truth." value={s ? (Object.entries(s.recommendation_validation.judge_verdicts).map(([k, v]) => `${k.replace('_', ' ')}: ${v}`).join(' · ') || NOT_EVALUATED) : NOT_EVALUATED} sub={s ? `${s.recommendation_validation.unsupported_numeric_claims_total} unsupported numeric claims flagged` : undefined} />
      </div>

      <h2 className="font-display mb-3 mt-9 text-lg font-semibold text-ink">Yield prediction model (held-out test years, independent of LLM metrics)</h2>
      {ym ? (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <M label="MAE" tip={EXPLAIN.mae} value={metric(yTest?.mae, 2)} sub={`full features · baseline ${metric(ym.comparison.full.baseline_crop_median.test?.mae, 2)}`} />
            <M label="RMSE" tip={EXPLAIN.rmse} value={metric(yTest?.rmse, 2)} sub={`baseline ${metric(ym.comparison.full.baseline_crop_median.test?.rmse, 2)}`} />
            <M label="R²" tip={EXPLAIN.r2} value={metric(yTest?.r2, 3)} sub={`baseline ${metric(ym.comparison.full.baseline_crop_median.test?.r2, 3)}`} />
            <M label="R² (log scale)" tip="R² on log1p(yield): not dominated by the largest-magnitude crops." value={metric(yTest?.r2_log1p, 3)} sub={`baseline ${metric(ym.comparison.full.baseline_crop_median.test?.r2_log1p, 3)}`} />
          </div>
          <Card className="mt-3">
            <p className="mb-2 text-xs text-mute">Split: {ym.split.strategy} · train {ym.split.n_train}, validation {ym.split.n_val}, test {ym.split.n_test}. Selected on validation RMSE (log scale), then refit on train+validation.</p>
            <Table dense rows={Object.entries(ym.comparison).flatMap(([fs, c]) => Object.entries(c).filter(([n]) => n !== 'selected').map(([n, d]) => ({ fs, n, val: d.val, test: d.test, selected: c.selected.name === n })))} rowKey={(x) => x.fs + x.n}
              columns={[{ key: 'fs', label: 'Feature set' }, { key: 'n', label: 'Model', render: (x) => <span>{x.n} {x.selected && <Badge tone="green">selected</Badge>}</span> },
                { key: 'v1', label: 'Val MAE', align: 'right', render: (x) => fmt(x.val.mae) }, { key: 'v2', label: 'Val RMSE (log)', align: 'right', render: (x) => fmt(x.val.rmse_log1p, 3) },
                { key: 't1', label: 'Test MAE', align: 'right', render: (x) => (x.test ? fmt(x.test.mae) : '—') }, { key: 't2', label: 'Test RMSE', align: 'right', render: (x) => (x.test ? fmt(x.test.rmse) : '—') }, { key: 't3', label: 'Test R²', align: 'right', render: (x) => (x.test ? fmt(x.test.r2, 3) : '—') }]} />
            {yCore && <p className="mt-2 text-[11px] text-dim">Core feature set (no area/inputs) test MAE {fmt(yCore.mae)}, RMSE {fmt(yCore.rmse)}, R² {fmt(yCore.r2, 3)}. 10–90% interval empirical coverage on test years: {fmt(ym.comparison.full.selected.interval_10_90.empirical_test_coverage * 100, 0)}% (nominal 80%).</p>}
          </Card>
        </>
      ) : <EmptyState title="Not evaluated" hint="Run python scripts/train_yield_model.py" />}

      {cmpId && cmp.data && s && (
        <Card className="mt-6">
          <CardHeader title="Experiment comparison" subtitle={`${runId} vs ${cmpId}`} />
          <Table dense rows={compareRows(r, cmp.data)} rowKey={(x) => x.m} columns={[{ key: 'm', label: 'Metric' }, { key: 'a', label: runId, align: 'right' }, { key: 'b', label: cmpId, align: 'right' }, { key: 'd', label: 'Δ (A − B)', align: 'right' }]} />
        </Card>
      )}

      {r && <TestCases run={r} />}

      <Card className="mt-6">
        <CardHeader title="What do these metrics mean?" icon={FlaskConical} />
        <dl className="grid gap-3 text-xs text-mute md:grid-cols-2">
          {Object.entries({ Faithfulness: EXPLAIN.faithfulness, 'Answer relevancy': EXPLAIN.answer_relevancy, 'Contextual precision': EXPLAIN.contextual_precision, 'Contextual recall': EXPLAIN.contextual_recall, 'Precision@K': EXPLAIN.precision, 'Recall@K': EXPLAIN.recall, MRR: EXPLAIN.mrr, 'Hit rate@K': EXPLAIN.hit, MAE: EXPLAIN.mae, RMSE: EXPLAIN.rmse, 'R²': EXPLAIN.r2 }).map(([k, v]) => <div key={k}><dt className="font-semibold text-ink">{k}</dt><dd>{v}</dd></div>)}
        </dl>
        <p className="mt-3 text-[11px] text-dim">Relevance labels come from deterministic filters over the real dataset (e.g. every record with the stated crop+state+season). Test queries are templates over real values and are not historical records. Metrics are intentionally not combined into one score.</p>
      </Card>
    </div>
  )
}

function compareRows(a, b) {
  const g = (r, path) => path.split('.').reduce((o, k) => o?.[k], r.summary)
  const defs = [['Precision@K', 'retrieval.precision_at_k'], ['Recall@K', 'retrieval.recall_at_k'], ['MRR', 'retrieval.mrr'], ['Hit rate@K', 'retrieval.hit_rate_at_k'],
    ['Faithfulness', 'deepeval.faithfulness.mean'], ['Answer relevancy', 'deepeval.answer_relevancy.mean'], ['Contextual precision', 'deepeval.contextual_precision.mean'], ['Contextual recall', 'deepeval.contextual_recall.mean'],
    ['Constraint satisfaction', 'recommendation_validation.constraint_satisfaction_rate'], ['Tests passed', 'n_passed'], ['Tests total', 'n_test_cases']]
  return defs.map(([m, p]) => {
    const x = g(a, p), y = g(b, p)
    return { m, a: x == null ? NOT_EVALUATED : fmt(x, 3), b: y == null ? NOT_EVALUATED : fmt(y, 3), d: x == null || y == null ? '—' : fmt(x - y, 3) }
  })
}

function TestCases({ run }) {
  const [type, setType] = useState('')
  const [status, setStatus] = useState('')
  const [sel, setSel] = useState(null)
  const rows = run.results.filter((x) => (!type || x.type === type) && (!status || (status === 'pass' ? x.passed : !x.passed)))
  return (
    <Card className="mt-6">
      <CardHeader title="Test-case explorer" subtitle={`${run.results.length} cases in ${run.run_id}`} action={
        <div className="flex gap-2"><Select value={type} onChange={(e) => setType(e.target.value)} aria-label="Type"><option value="">All types</option>{[...new Set(run.results.map((x) => x.type))].map((t) => <option key={t}>{t}</option>)}</Select>
          <Select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Status"><option value="">Pass + fail</option><option value="pass">Passed</option><option value="fail">Failed</option></Select></div>} />
      <Table rows={rows} rowKey={(x) => x.test_case_id} onRow={setSel} columns={[
        { key: 'id', label: 'ID', render: (x) => <span className="font-mono text-xs">{x.test_case_id}</span> },
        { key: 'type', label: 'Type' },
        { key: 'input', label: 'Query', render: (x) => <span className="line-clamp-1 max-w-sm">{x.input}</span> },
        { key: 'p', label: 'Result', render: (x) => (x.passed ? <Badge tone="green"><CheckCircle2 className="h-3 w-3" /> pass</Badge> : <Badge tone="red"><XCircle className="h-3 w-3" /> fail</Badge>) },
        { key: 'pr', label: 'P@K', align: 'right', render: (x) => (x.retrieval_metrics ? fmt(x.retrieval_metrics.precision_at_k, 2) : '—') },
        { key: 'f', label: 'Failure reasons', render: (x) => <span className="text-xs text-red-300">{x.failure_reasons.join('; ')}</span> },
      ]} />
      <Drawer open={!!sel} onClose={() => setSel(null)} title={sel?.test_case_id || ''}>
        {sel && (
          <div className="space-y-3 text-sm">
            <Badge tone={sel.passed ? 'green' : 'red'}>{sel.passed ? 'passed' : 'failed'}</Badge> <Badge>{sel.type}</Badge>
            <div><p className="text-xs font-semibold">Input</p><p>{sel.input}</p></div>
            <div><p className="text-xs font-semibold">Expected</p><pre className="overflow-x-auto rounded-lg bg-glass-2 p-2 text-[11px]">{JSON.stringify({ ...sel.expected, allowed_crops: sel.expected.allowed_crops ? `${sel.expected.allowed_crops.length} crops` : undefined }, null, 1)}</pre><p className="mt-1 text-xs text-mute">Reference: {sel.reference_answer}</p><p className="text-[11px] text-dim">Label provenance: {sel.provenance}</p></div>
            <div><p className="text-xs font-semibold">Generated answer</p><p className="whitespace-pre-wrap rounded-lg bg-glass-2 p-2 text-xs">{sel.generated_answer || 'Not generated for this case (retrieval-only test).'}</p></div>
            <div><p className="text-xs font-semibold">Retrieved case IDs vs expected relevant ({sel.n_expected_relevant})</p>
              <div className="flex flex-wrap gap-1">{sel.retrieved_case_ids.map((id) => <Badge key={id} tone={sel.expected_relevant_case_ids.includes(id) ? 'green' : 'neutral'}><span className="font-mono">{id}</span></Badge>)}</div></div>
            {sel.retrieval_metrics && <KV items={[['Precision@K', fmt(sel.retrieval_metrics.precision_at_k, 3)], ['Recall@K', fmt(sel.retrieval_metrics.recall_at_k, 3)], ['MRR', fmt(sel.retrieval_metrics.mrr, 3)], ['Hit@K', fmt(sel.retrieval_metrics.hit_rate_at_k, 3)]]} />}
            {sel.deepeval && <div><p className="text-xs font-semibold">DeepEval breakdown</p><Table dense rows={Object.entries(sel.deepeval).map(([k, v]) => ({ k, ...v }))} rowKey={(x) => x.k} columns={[{ key: 'k', label: 'Metric' }, { key: 's', label: 'Score', render: (x) => (x.available ? fmt(x.score, 3) : NOT_EVALUATED) }, { key: 'r', label: 'Reason', render: (x) => <span className="text-[11px]">{x.reason}</span> }]} /></div>}
            {sel.recommendation_checks && <div><p className="text-xs font-semibold">Recommendation checks</p><pre className="overflow-x-auto rounded-lg bg-glass-2 p-2 text-[11px]">{JSON.stringify(sel.recommendation_checks, null, 1)}</pre></div>}
            {sel.judge_verdict && <p className="text-xs">Judge: <b>{sel.judge_verdict.verdict}</b> ({sel.judge_verdict.llm_status})</p>}
            {sel.failure_reasons.length > 0 && <div><p className="text-xs font-semibold text-red-300">Failure reasons</p><ul className="list-disc pl-5 text-xs text-red-300">{sel.failure_reasons.map((f) => <li key={f}>{f}</li>)}</ul></div>}
          </div>
        )}
      </Drawer>
    </Card>
  )
}
