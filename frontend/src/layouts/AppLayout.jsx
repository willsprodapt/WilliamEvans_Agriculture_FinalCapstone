import { useState } from 'react'
import { Link, NavLink, Outlet } from 'react-router-dom'
import { Bell, Menu, Sprout, X } from 'lucide-react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { cx } from '../components/ui'

const NAV = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/advisory', label: 'Crop Advisory' },
  { to: '/cases', label: 'Case Explorer' },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/insights', label: 'Insights & Feedback' },
]

function StatusPill() {
  const { data, error } = useApi(() => api.system.status(), [])
  const items = data && [
    ['Data', data.dataset?.available], ['Index', data.index?.valid], ['Model', data.yield_model?.available], ['Claude', data.llm?.configured],
  ]
  const allOk = items?.every(([, ok]) => ok)
  return (
    <div className="group relative">
      <button className="glass flex items-center gap-2 rounded-full px-3 py-1.5 text-xs text-mute" aria-label="System status">
        <span className={cx('h-2 w-2 rounded-full', error ? 'bg-red-400' : !items ? 'bg-white/30' : allOk ? 'bg-accent' : 'bg-amber-400')} />
        <span className="hidden sm:inline">{error ? 'API offline' : !items ? 'Checking' : allOk ? 'All systems ready' : 'Degraded'}</span>
      </button>
      {items && (
        <div className="glass invisible absolute right-0 top-full z-40 mt-2 w-52 rounded-2xl bg-deep/95 p-3 opacity-0 transition group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100">
          {items.map(([l, ok]) => (
            <div key={l} className="flex items-center justify-between py-1 text-xs"><span className="text-mute">{l}</span><span className={ok ? 'text-accent' : 'text-amber-300'}>{ok ? 'ready' : 'unavailable'}</span></div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function AppLayout() {
  const [open, setOpen] = useState(false)
  return (
    <div className="min-h-screen px-3 pb-10 pt-3 sm:px-6">
      <header className="glass sticky top-3 z-30 mx-auto flex max-w-[1240px] items-center justify-between gap-3 rounded-full px-4 py-2.5 sm:px-5">
        <Link to="/" className="flex items-center gap-2" aria-label="TerraMind AI home">
          <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-gradient-to-br from-[#d6f984] to-emerald-400 text-deep shadow-[0_6px_18px_-6px_rgba(196,242,106,.7)]"><Sprout className="h-[18px] w-[18px]" aria-hidden /></span>
          <span className="font-display text-[19px] font-semibold tracking-tight text-ink">Terra<span className="text-gradient">Mind</span> AI</span>
        </Link>
        <nav aria-label="Main" className="hidden items-center gap-1 lg:flex">
          {NAV.map(({ to, label, end }) => (
            <NavLink key={to} to={to} end={end}
              className={({ isActive }) => cx('rounded-full px-4 py-1.5 text-[13px] transition-colors', isActive ? 'bg-accent/15 font-semibold text-accent ring-1 ring-accent/30' : 'text-mute hover:bg-white/5 hover:text-ink')}>
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <StatusPill />
          <Link to="/insights" className="glass hidden h-8 w-8 items-center justify-center rounded-full text-mute hover:text-ink sm:flex" aria-label="Feedback and insights"><Bell className="h-4 w-4" /></Link>
          <button onClick={() => setOpen(!open)} aria-label="Toggle navigation" aria-expanded={open} className="glass flex h-9 w-9 items-center justify-center rounded-full lg:hidden">
            {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
        </div>
      </header>
      {open && (
        <nav aria-label="Mobile" className="glass fade-up sticky top-[72px] z-30 mx-auto mt-2 max-w-[1240px] rounded-3xl bg-deep/95 p-2 lg:hidden">
          {NAV.map(({ to, label, end }) => (
            <NavLink key={to} to={to} end={end} onClick={() => setOpen(false)}
              className={({ isActive }) => cx('block rounded-2xl px-4 py-3 text-sm', isActive ? 'bg-white/10 text-ink' : 'text-mute')}>{label}</NavLink>
          ))}
        </nav>
      )}
      <main className="mx-auto mt-10 max-w-[1240px] px-1 sm:px-2">
        <Outlet />
        <footer className="mt-16 flex flex-col items-center gap-1 border-t border-line pt-6 text-center text-[11px] text-dim">
          <span className="font-display text-xs text-mute">Terra<span className="text-accent">Mind</span> AI · Smarter crop decisions. Evidence-driven farming.</span>
          <span>Decision support only: recommendations summarise historical records and model estimates and do not guarantee yields.</span>
        </footer>
      </main>
    </div>
  )
}
