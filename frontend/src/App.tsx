/** Route table. The 3D scenes are lazy-loaded per route, so the initial bundle never
 * includes the three.js chunk. */
import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Skeleton } from './components/ui'
import { AnalyzePage } from './pages/AnalyzePage'

const ResultsPage = lazy(() => import('./pages/ResultsPage').then((m) => ({ default: m.ResultsPage })))
const HistoryPage = lazy(() => import('./pages/HistoryPage').then((m) => ({ default: m.HistoryPage })))
const EvaluationPage = lazy(() => import('./pages/EvaluationPage').then((m) => ({ default: m.EvaluationPage })))
const AboutPage = lazy(() => import('./pages/AboutPage').then((m) => ({ default: m.AboutPage })))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage').then((m) => ({ default: m.NotFoundPage })))

export function App() {
  return (
    <Layout>
      <Suspense fallback={<Skeleton lines={6} />}>
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/results/:id" element={<ResultsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}
