// Dedicated API service layer. No mock fallbacks: failures are surfaced to the UI as ApiError.
const BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, kind = 'http' } = {}) {
    super(message)
    this.status = status
    this.detail = detail
    this.kind = kind // 'http' | 'network' | 'timeout'
  }
}

function describeDetail(detail) {
  if (!detail) return null
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d) => `${(d.loc || []).filter((x) => x !== 'body' && x !== 'query').join('.') || 'input'}: ${d.msg}`)
      .join('; ')
  }
  return JSON.stringify(detail)
}

async function request(path, { method = 'GET', body, params, timeout = 30000, signal } = {}) {
  const url = new URL(BASE + path)
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v === undefined || v === null || v === '') return
      if (Array.isArray(v)) v.forEach((x) => url.searchParams.append(k, x))
      else url.searchParams.set(k, v)
    })
  }
  const ctl = new AbortController()
  const timer = setTimeout(() => ctl.abort('timeout'), timeout)
  if (signal) signal.addEventListener('abort', () => ctl.abort('cancelled'))
  try {
    const res = await fetch(url, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: ctl.signal,
    })
    let data = null
    const text = await res.text()
    try { data = text ? JSON.parse(text) : null } catch { data = text }
    if (!res.ok) {
      const detail = describeDetail(data?.detail ?? data)
      throw new ApiError(detail || `Request failed (${res.status})`, { status: res.status, detail: data?.detail, kind: 'http' })
    }
    return data
  } catch (e) {
    if (e instanceof ApiError) throw e
    if (ctl.signal.aborted && ctl.signal.reason === 'timeout') throw new ApiError('The request timed out. The server may be busy.', { kind: 'timeout' })
    if (ctl.signal.aborted) throw new ApiError('Request cancelled', { kind: 'cancelled' })
    throw new ApiError(`Cannot reach the TerraMind API at ${BASE}. Is the backend running?`, { kind: 'network' })
  } finally {
    clearTimeout(timer)
  }
}

export const api = {
  base: BASE,
  system: {
    status: () => request('/api/system/status'),
    stats: () => request('/api/stats'),
  },
  advisory: {
    startJob: (payload) => request('/api/advisory/jobs', { method: 'POST', body: payload }),
    job: (id) => request(`/api/advisory/jobs/${id}`),
    query: (payload) => request('/api/advisory/query', { method: 'POST', body: payload, timeout: 180000 }),
    classify: (payload) => request('/api/advisory/classify', { method: 'POST', body: payload, timeout: 60000 }),
    recent: (limit = 8) => request('/api/advisory/recent', { params: { limit } }),
    get: (id) => request(`/api/advisory/queries/${id}`),
    summary: () => request('/api/insights/advisory-summary'),
  },
  cases: {
    list: (params) => request('/api/cases', { params }),
    get: (id) => request(`/api/cases/${id}`),
    search: (payload) => request('/api/cases/search', { method: 'POST', body: payload, timeout: 60000 }),
    options: () => request('/api/metadata/options'),
  },
  predictions: {
    yield: (payload) => request('/api/predictions/yield', { method: 'POST', body: payload }),
    suitability: (payload) => request('/api/predictions/suitability', { method: 'POST', body: payload }),
    impact: (payload) => request('/api/analysis/impact', { method: 'POST', body: payload }),
  },
  evaluation: {
    overview: () => request('/api/evaluations'),
    runs: () => request('/api/evaluations/runs'),
    run: (id) => request(`/api/evaluations/runs/${id}`, { timeout: 60000 }),
    testCases: () => request('/api/evaluations/test-cases'),
    start: (params) => request('/api/evaluations/run', { method: 'POST', params }),
  },
  feedback: {
    submit: (payload) => request('/api/feedback', { method: 'POST', body: payload }),
    list: (params) => request('/api/feedback', { params }),
    summary: () => request('/api/feedback/summary'),
  },
}
