// src/pages/History.tsx
import { useState, useEffect } from 'react'
import { historyAPI, HistoryRun } from '../lib/api'
import { Clock, RefreshCw } from 'lucide-react'

function complianceColor(score: number | null): string {
  if (score === null) return 'var(--text-muted)'
  if (score >= 85) return 'var(--low)'
  if (score >= 65) return 'var(--medium)'
  if (score >= 40) return 'var(--high)'
  return 'var(--critical)'
}

export default function History() {
  const [runs, setRuns]       = useState<HistoryRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)

  const load = () => {
    setLoading(true); setError(null)
    historyAPI(100)
      .then(setRuns)
      .catch(() => setError('Could not load history from MLflow.'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div className="page-title">Audit History</div>
          <div className="page-sub">Compliance audit runs tracked in MLflow</div>
        </div>
        <button className="btn btn-ghost" onClick={load} style={{ fontSize: 12 }}>
          <RefreshCw size={12} /> Refresh
        </button>
      </div>

      {error && (
        <div style={{ background: '#3b0e0e', border: '1px solid var(--critical)', borderRadius: 8, padding: 14, marginBottom: 16, color: 'var(--critical)', fontSize: 13 }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="empty-state"><span className="spinner" /> Loading…</div>
      ) : runs.length === 0 ? (
        <div className="empty-state">
          <Clock size={32} />
          No audit runs recorded yet — run a compliance query in Audit
        </div>
      ) : (
        <div className="card">
          <table className="gaps-table">
            <thead>
              <tr><th>When</th><th>Document</th><th>Query</th><th>Score</th><th>Gaps</th><th>Conflicts</th><th>Frameworks</th></tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {r.start_time ? new Date(Number(r.start_time)).toLocaleString() : '—'}
                  </td>
                  <td style={{ fontSize: 12 }}>{r.policy_name || '—'}</td>
                  <td style={{ maxWidth: 320, fontSize: 12, color: 'var(--text-dim)' }}>{r.query || '—'}</td>
                  <td style={{ fontWeight: 700, color: complianceColor(r.compliance_score) }}>
                    {r.compliance_score != null ? `${r.compliance_score.toFixed(0)}/100` : '—'}
                  </td>
                  <td>{r.total_gaps ?? '—'}</td>
                  <td>{r.total_conflicts ?? '—'}</td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{r.jurisdictions || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
