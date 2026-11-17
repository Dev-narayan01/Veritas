/** Custom 404. */
import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <div className="flex flex-col items-center gap-3 py-24 text-center">
      <p className="font-mono text-5xl">404</p>
      <p className="muted">That page does not exist.</p>
      <Link className="btn btn-primary" to="/">Back to Analyze</Link>
    </div>
  )
}
