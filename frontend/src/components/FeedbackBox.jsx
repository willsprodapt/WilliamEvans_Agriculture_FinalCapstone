import { useState } from 'react'
import { CheckCircle2, ThumbsDown, ThumbsUp } from 'lucide-react'
import { api } from '../services/api'
import { Button, Field, Select, Textarea, cx } from './ui'

const CATEGORIES = [
  ['incorrect_crop', 'Incorrect crop'], ['missing_evidence', 'Missing evidence'], ['wrong_conditions', 'Conditions not matched'],
  ['unclear_explanation', 'Unclear explanation'], ['yield_prediction', 'Yield prediction'], ['risk_assessment', 'Risk assessment'], ['other', 'Other'],
]

export default function FeedbackBox({ queryId, crop, recommendationId, caseIds = [], compact = false }) {
  const [helpful, setHelpful] = useState(null)
  const [rating, setRating] = useState('')
  const [category, setCategory] = useState('')
  const [comment, setComment] = useState('')
  const [state, setState] = useState({ busy: false, done: false, error: null })

  async function submit() {
    if (helpful === null) return
    setState({ busy: true, done: false, error: null })
    try {
      await api.feedback.submit({
        query_id: queryId, crop: crop || null, recommendation_id: recommendationId || null, helpful,
        rating: rating ? Number(rating) : null, category: category || null, comment: comment.trim() || null, case_ids: caseIds.slice(0, 20),
      })
      setState({ busy: false, done: true, error: null })
    } catch (e) {
      setState({ busy: false, done: false, error: e.message })
    }
  }

  if (state.done) {
    return <p className="flex items-center gap-2 rounded-xl bg-accent/10 px-3 py-2 text-xs text-ink"><CheckCircle2 className="h-4 w-4" /> Thanks — feedback saved{crop ? ` for ${crop}` : ''}. It is stored for human review; it does not retrain the system automatically.</p>
  }
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {!compact && <span className="text-xs font-medium text-mute">Was this {crop ? `${crop} recommendation` : 'advice'} helpful?</span>}
        {[[true, 'Helpful', ThumbsUp], [false, 'Not helpful', ThumbsDown]].map(([v, l, I]) => (
          <button key={l} onClick={() => setHelpful(v)} aria-pressed={helpful === v}
            className={cx('inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-medium', helpful === v ? (v ? 'border-emerald-600 bg-accent/10 text-ink' : 'border-amber-500 bg-amber-400/10 text-amber-200') : 'border-line bg-glass text-mute hover:bg-glass-2')}>
            <I className="h-3.5 w-3.5" aria-hidden /> {l}
          </button>
        ))}
      </div>
      {helpful !== null && (
        <div className="fade-up grid gap-2 sm:grid-cols-2">
          <Field label="Rating (optional)"><Select value={rating} onChange={(e) => setRating(e.target.value)} aria-label="Rating"><option value="">—</option>{[5, 4, 3, 2, 1].map((n) => <option key={n} value={n}>{n} / 5</option>)}</Select></Field>
          {!helpful && <Field label="What went wrong?"><Select value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Category"><option value="">—</option>{CATEGORIES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</Select></Field>}
          <Field label="Comments (optional)" className="sm:col-span-2"><Textarea rows={2} maxLength={2000} value={comment} onChange={(e) => setComment(e.target.value)} placeholder="Tell us more" /></Field>
          <div className="sm:col-span-2 flex items-center gap-3">
            <Button size="sm" loading={state.busy} onClick={submit}>Submit feedback</Button>
            {state.error && <span role="alert" className="text-xs text-red-300">{state.error}</span>}
          </div>
        </div>
      )}
    </div>
  )
}
