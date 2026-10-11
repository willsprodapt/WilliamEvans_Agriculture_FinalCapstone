import { Route, Routes } from 'react-router-dom'
import AppLayout from './layouts/AppLayout'
import Dashboard from './pages/Dashboard'
import Advisory from './pages/Advisory'
import CaseExplorer from './pages/CaseExplorer'
import Evaluation from './pages/Evaluation'
import Insights from './pages/Insights'

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="advisory" element={<Advisory />} />
        <Route path="cases" element={<CaseExplorer />} />
        <Route path="evaluation" element={<Evaluation />} />
        <Route path="insights" element={<Insights />} />
        <Route path="*" element={<p className="p-6 text-sm">Page not found.</p>} />
      </Route>
    </Routes>
  )
}
