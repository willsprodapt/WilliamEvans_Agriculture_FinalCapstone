import { useCallback, useEffect, useRef, useState } from 'react'

// Loads data from an API call; exposes loading/error/reload. Never substitutes mock data.
export function useApi(fn, deps = [], { immediate = true } = {}) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(immediate)
  const alive = useRef(true)
  const fnRef = useRef(fn)
  fnRef.current = fn

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const r = await fnRef.current()
      if (alive.current) setData(r)
      return r
    } catch (e) {
      if (alive.current) setError(e)
      return null
    } finally {
      if (alive.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    alive.current = true
    if (immediate) load()
    return () => { alive.current = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, error, loading, reload: load, setData }
}
