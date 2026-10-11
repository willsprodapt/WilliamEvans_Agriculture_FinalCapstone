import { useState } from 'react'
import { AlertTriangle, CheckCircle2, ChevronDown, CircleHelp, Inbox, Loader2, XCircle } from 'lucide-react'

export const cx = (...a) => a.filter(Boolean).join(' ')

export function Card({ children, className = '', dark = false, as: Tag = 'section', ...rest }) {
  return (
    <Tag
      className={cx('glass rounded-3xl p-6', dark && '!border-accent/30 !bg-gradient-to-br !from-accent/20 !to-emerald-500/5', className)}
      {...rest}
    >
      {children}
    </Tag>
  )
}

export function CardHeader({ title, subtitle, action, icon: Icon }) {
  return (
    <div className="mb-5 flex items-start justify-between gap-3">
      <div className="flex items-start gap-3">
        {Icon && <span className="mt-0.5 flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent/25 to-emerald-400/10 text-accent"><Icon className="h-4 w-4" aria-hidden /></span>}
        <div>
          <h2 className="text-[16px] font-semibold text-ink">{title}</h2>
          {subtitle && <p className="mt-0.5 text-xs leading-relaxed text-mute">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  )
}

const BTN = {
  primary: 'bg-gradient-to-b from-[#d6f984] to-[#a9e046] text-deep hover:brightness-105 shadow-[0_8px_26px_-8px_rgba(196,242,106,0.65)] disabled:bg-white/20 disabled:text-white/40 disabled:shadow-none',
  secondary: 'glass text-ink hover:bg-glass-3 disabled:text-dim',
  ghost: 'text-mute hover:bg-glass-3 hover:text-ink disabled:text-dim',
  danger: 'bg-red-500/80 text-white hover:bg-red-500',
}
export function Button({ variant = 'primary', size = 'md', loading, icon: Icon, children, className = '', ...rest }) {
  return (
    <button
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-full font-semibold transition-all active:scale-[.98] disabled:cursor-not-allowed',
        size === 'sm' ? 'px-3.5 py-1.5 text-xs' : 'px-5 py-2.5 text-sm',
        BTN[variant],
        className,
      )}
      disabled={loading || rest.disabled}
      {...rest}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : Icon ? <Icon className="h-4 w-4" aria-hidden /> : null}
      {children}
    </button>
  )
}

const BADGE = {
  neutral: 'bg-white/8 text-mute border-line',
  green: 'bg-accent/12 text-accent border-accent/25',
  amber: 'bg-amber-400/10 text-amber-200 border-amber-300/25',
  red: 'bg-red-400/10 text-red-300 border-red-300/25',
  blue: 'bg-sky-400/10 text-sky-200 border-sky-300/25',
  dark: 'bg-accent text-deep border-accent',
}
export function Badge({ tone = 'neutral', children, className = '', title }) {
  return (
    <span title={title} className={cx('inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-medium', BADGE[tone], className)}>
      {children}
    </span>
  )
}

export function Field({ label, hint, children, htmlFor, className = '' }) {
  return (
    <div className={className}>
      <label htmlFor={htmlFor} className="mb-1.5 block text-xs font-medium text-mute">{label}</label>
      {children}
      {hint && <p className="mt-1 text-[11px] text-dim">{hint}</p>}
    </div>
  )
}

const inputCls =
  'w-full rounded-2xl border border-line bg-black/20 px-3.5 py-2.5 text-sm text-ink placeholder:text-dim focus:border-accent/60 disabled:opacity-50'
export function Input({ className, ...props }) { return <input {...props} className={cx(inputCls, className)} /> }
export function Select({ children, className, ...props }) {
  return <select {...props} className={cx(inputCls, 'pr-8', className)}>{children}</select>
}
export function Textarea({ className, ...props }) { return <textarea {...props} className={cx(inputCls, 'resize-y', className)} /> }

export function Spinner({ label = 'Loading' }) {
  return (
    <div role="status" className="flex items-center gap-2 py-6 text-sm text-mute">
      <Loader2 className="h-4 w-4 animate-spin text-accent" aria-hidden /> {label}…
    </div>
  )
}

export function Skeleton({ className = 'h-24' }) {
  return <div className={cx('shimmer rounded-2xl', className)} aria-hidden />
}

export function ErrorState({ error, onRetry, title = 'Something went wrong' }) {
  if (!error) return null
  const offline = error.kind === 'network'
  return (
    <div role="alert" className="rounded-2xl border border-red-300/30 bg-red-400/10 p-4 text-sm text-red-200">
      <div className="flex items-start gap-2">
        <XCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
        <div className="min-w-0">
          <p className="font-medium">{offline ? 'Backend unreachable' : title}</p>
          <p className="mt-0.5 break-words text-red-300">{error.message}</p>
          {error.status === 503 && <p className="mt-1 text-xs text-red-300/80">A required artifact is missing. Check the README setup steps (prepare dataset, build index, train model).</p>}
          {onRetry && <button onClick={onRetry} className="mt-2 text-xs font-medium underline">Try again</button>}
        </div>
      </div>
    </div>
  )
}

export function EmptyState({ title = 'Nothing to show', hint, icon: Icon = Inbox, children }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 rounded-2xl border border-dashed border-line px-4 py-10 text-center">
      <Icon className="h-6 w-6 text-dim" aria-hidden />
      <p className="text-sm font-medium text-ink">{title}</p>
      {hint && <p className="max-w-md text-xs text-mute">{hint}</p>}
      {children}
    </div>
  )
}

export function Notice({ tone = 'amber', children, title }) {
  const t = { amber: 'border-amber-300/25 bg-amber-400/10 text-amber-100', blue: 'border-sky-300/25 bg-sky-400/10 text-sky-100', green: 'border-accent/25 bg-accent/10 text-ink', red: 'border-red-300/30 bg-red-400/10 text-red-200' }[tone]
  const I = tone === 'green' ? CheckCircle2 : AlertTriangle
  return (
    <div className={cx('flex gap-2.5 rounded-2xl border p-3.5 text-xs leading-relaxed', t)} role={tone === 'red' ? 'alert' : 'note'}>
      <I className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
      <div>{title && <p className="font-semibold">{title}</p>}<div>{children}</div></div>
    </div>
  )
}

export function Tooltip({ text, children }) {
  return (
    <span className="group relative inline-flex">
      {children || <CircleHelp className="h-3.5 w-3.5 text-dim" aria-label="info" />}
      <span role="tooltip" className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-1.5 hidden w-60 -translate-x-1/2 rounded-xl border border-line bg-deep px-3 py-2 text-[11px] font-normal leading-snug text-ink shadow-xl group-hover:block group-focus-within:block">
        {text}
      </span>
    </span>
  )
}

export function Stat({ label, value, sub, dark = false, icon: Icon }) {
  return (
    <Card dark={dark} className="lift relative flex flex-col justify-between overflow-hidden !p-5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium uppercase tracking-wider text-mute">{label}</p>
        {Icon && <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-accent/12 text-accent"><Icon className="h-4 w-4" aria-hidden /></span>}
      </div>
      <p className={cx('font-display mt-4 text-4xl font-semibold tabular-nums tracking-tight', dark ? 'text-gradient' : 'text-ink')}>{value}</p>
      {sub && <p className="mt-1.5 text-xs text-dim">{sub}</p>}
    </Card>
  )
}

export function Table({ columns, rows, onRow, empty = 'No rows', rowKey, dense }) {
  if (!rows?.length) return <EmptyState title={empty} />
  return (
    <div className="scroll-thin overflow-x-auto rounded-2xl border border-line">
      <table className="w-full min-w-max text-left text-sm">
        <thead className="bg-white/5 text-[11px] uppercase tracking-wide text-mute">
          <tr>{columns.map((c) => <th key={c.key} scope="col" className={cx('px-3.5 py-2.5 font-medium', c.align === 'right' && 'text-right')}>{c.label}</th>)}</tr>
        </thead>
        <tbody className="divide-y divide-white/6">
          {rows.map((r, i) => (
            <tr
              key={rowKey ? rowKey(r) : i}
              onClick={onRow ? () => onRow(r) : undefined}
              onKeyDown={onRow ? (e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onRow(r)) : undefined}
              tabIndex={onRow ? 0 : undefined}
              className={cx(onRow && 'cursor-pointer hover:bg-white/6 focus-visible:bg-white/6')}
            >
              {columns.map((c) => (
                <td key={c.key} className={cx('px-3.5 align-top text-ink/90', dense ? 'py-2' : 'py-3', c.align === 'right' && 'text-right tabular-nums')}>
                  {c.render ? c.render(r) : r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Collapsible({ title, children, defaultOpen = false, right }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-2xl border border-line bg-white/[0.03]">
      <button onClick={() => setOpen(!open)} aria-expanded={open} className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left text-sm font-medium text-ink">
        <span>{title}</span>
        <span className="flex items-center gap-2 text-mute">{right}<ChevronDown className={cx('h-4 w-4 transition-transform', open && 'rotate-180')} aria-hidden /></span>
      </button>
      {open && <div className="border-t border-line p-4">{children}</div>}
    </div>
  )
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div role="tablist" className="glass inline-flex gap-1 rounded-full p-1">
      {tabs.map((t) => (
        <button key={t.value} role="tab" aria-selected={value === t.value} onClick={() => onChange(t.value)}
          className={cx('rounded-full px-4 py-1.5 text-xs font-medium transition-colors', value === t.value ? 'bg-accent text-deep' : 'text-mute hover:text-ink')}>
          {t.label}
        </button>
      ))}
    </div>
  )
}

export function Drawer({ open, onClose, title, children }) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label={title}>
      <div className="absolute inset-0 bg-black/55 backdrop-blur-sm" onClick={onClose} />
      <div className="fade-up relative h-full w-full max-w-lg overflow-y-auto border-l border-line bg-[#0a2013]/95 p-6 shadow-2xl backdrop-blur-xl">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-base font-semibold text-ink">{title}</h2>
          <Button variant="secondary" size="sm" onClick={onClose}>Close</Button>
        </div>
        {children}
      </div>
    </div>
  )
}

export function KV({ items }) {
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm sm:grid-cols-3">
      {items.filter(Boolean).map(([k, v]) => (
        <div key={k} className="min-w-0"><dt className="text-[11px] text-dim">{k}</dt><dd className="break-words font-medium text-ink tabular-nums">{v ?? '—'}</dd></div>
      ))}
    </dl>
  )
}

export function ProgressBar({ value, tone = 'green' }) {
  const pct = Math.max(0, Math.min(1, value || 0)) * 100
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
      <div className={cx('h-full rounded-full', tone === 'amber' ? 'bg-amber-400' : 'bg-accent')} style={{ width: `${pct}%` }} />
    </div>
  )
}

export function PageHeader({ title, subtitle, actions, eyebrow }) {
  return (
    <div className="mb-9 flex flex-wrap items-end justify-between gap-5">
      <div className="min-w-0">
        {eyebrow && <p className="eyebrow mb-3">{eyebrow}</p>}
        <h1 className="font-display text-4xl font-semibold tracking-tight text-ink sm:text-5xl">{title}</h1>
        {subtitle && <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-mute">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2.5">{actions}</div>}
    </div>
  )
}
