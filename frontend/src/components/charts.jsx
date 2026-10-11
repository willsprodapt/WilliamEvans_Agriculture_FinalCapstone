import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip as RTooltip, XAxis, YAxis } from 'recharts'
import { CHART, fmt } from '../utils/format'
import { EmptyState } from './ui'

const axis = { fontSize: 11, fill: '#a4bfab' }
const tip = { contentStyle: { borderRadius: 12, border: '1px solid rgba(255,255,255,0.15)', background: 'rgba(8,28,17,0.92)', color: '#eaf4ec', fontSize: 12 }, cursor: { fill: 'rgba(255,255,255,0.05)' } }

export function HBar({ data, x = 'value', y = 'name', height = 260, color = '#c4f26a', signed = false, fmtX }) {
  if (!data?.length) return <EmptyState title="No data" />
  return (
    <div style={{ height }} role="img" aria-label="Bar chart">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
          <defs>
            <linearGradient id="gH" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor="#34d399" stopOpacity={0.85} /><stop offset="100%" stopColor="#c4f26a" /></linearGradient>
            <linearGradient id="gV" x1="0" y1="1" x2="0" y2="0"><stop offset="0%" stopColor="#34d399" stopOpacity={0.8} /><stop offset="100%" stopColor="#c4f26a" /></linearGradient>
          </defs><CartesianGrid horizontal={false} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 5" />
          <XAxis type="number" tick={axis} tickFormatter={fmtX} domain={signed ? [-1, 1] : undefined} />
          <YAxis type="category" dataKey={y} tick={axis} width={110} interval={0} />
          <RTooltip {...tip} formatter={(v) => fmt(v, 3)} />
          <Bar isAnimationActive={false} dataKey={x} radius={[0, 8, 8, 0]} fill={color === '#c4f26a' ? 'url(#gH)' : color} barSize={14}>
            {signed && data.map((d, i) => <Cell key={i} fill={d[x] >= 0 ? '#c4f26a' : '#f59e0b'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function VBar({ data, x = 'name', bars, height = 240, stacked = false }) {
  if (!data?.length) return <EmptyState title="No data" />
  return (
    <div style={{ height }} role="img" aria-label="Column chart">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ left: 0, right: 8, top: 4, bottom: 4 }}>
          <defs>
            <linearGradient id="gH" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor="#34d399" stopOpacity={0.85} /><stop offset="100%" stopColor="#c4f26a" /></linearGradient>
            <linearGradient id="gV" x1="0" y1="1" x2="0" y2="0"><stop offset="0%" stopColor="#34d399" stopOpacity={0.8} /><stop offset="100%" stopColor="#c4f26a" /></linearGradient>
          </defs><CartesianGrid vertical={false} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 5" />
          <XAxis dataKey={x} tick={axis} interval={0} angle={data.length > 6 ? -25 : 0} textAnchor={data.length > 6 ? 'end' : 'middle'} height={data.length > 6 ? 56 : 28} />
          <YAxis tick={axis} />
          <RTooltip {...tip} formatter={(v) => fmt(v, 3)} />
          {bars.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
          {bars.map((b, i) => <Bar isAnimationActive={false} key={b.key} dataKey={b.key} name={b.label || b.key} fill={b.color || (i === 0 ? 'url(#gV)' : CHART[i])} stackId={stacked ? 's' : undefined} radius={stacked ? 0 : [8, 8, 0, 0]} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function Donut({ data, height = 220 }) {
  if (!data?.length) return <EmptyState title="No data" />
  return (
    <div style={{ height }} role="img" aria-label="Donut chart">
      <ResponsiveContainer>
        <PieChart>
          <Pie isAnimationActive={false} data={data} dataKey="value" nameKey="name" innerRadius="55%" outerRadius="85%" paddingAngle={2}>
            {data.map((_, i) => <Cell key={i} fill={CHART[i % CHART.length]} />)}
          </Pie>
          <RTooltip {...tip} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

export function Lines({ data, x, lines, height = 220 }) {
  if (!data?.length) return <EmptyState title="No data" />
  return (
    <div style={{ height }} role="img" aria-label="Line chart">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ left: 0, right: 12, top: 4, bottom: 4 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.08)" />
          <XAxis dataKey={x} tick={axis} />
          <YAxis tick={axis} />
          <RTooltip {...tip} />
          {lines.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
          {lines.map((l, i) => <Line isAnimationActive={false} key={l.key} type="monotone" dataKey={l.key} name={l.label || l.key} stroke={l.color || CHART[i]} strokeWidth={2} dot={false} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export function Scatter2({ data, xKey, yKey, xLabel, yLabel, height = 240 }) {
  if (!data?.length) return <EmptyState title="No data" />
  return (
    <div style={{ height }} role="img" aria-label="Scatter chart">
      <ResponsiveContainer>
        <ScatterChart margin={{ left: 0, right: 12, top: 8, bottom: 16 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.08)" />
          <XAxis type="number" dataKey={xKey} name={xLabel} tick={axis} label={{ value: xLabel, position: 'insideBottom', offset: -8, fontSize: 11, fill: '#a4bfab' }} />
          <YAxis type="number" dataKey={yKey} name={yLabel} tick={axis} />
          <RTooltip {...tip} cursor={{ strokeDasharray: '3 3' }} />
          <Scatter isAnimationActive={false} data={data} fill="#34d399" fillOpacity={0.55} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
