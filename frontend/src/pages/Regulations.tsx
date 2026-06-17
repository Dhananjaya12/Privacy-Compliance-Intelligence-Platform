// src/pages/Regulations.tsx
import { useEffect, useState } from 'react'
import { regulationsAPI, regulationChunksAPI, RegulationChunk } from '../lib/api'
import { BookOpen } from 'lucide-react'

export default function Regulations() {
  const [regs, setRegs]       = useState<string[]>([])
  const [active, setActive]   = useState<string>('')
  const [chunks, setChunks]   = useState<RegulationChunk[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  useEffect(() => {
    regulationsAPI().then(rs => {
      setRegs(rs)
      if (rs.length) setActive(rs[0])
    }).catch(() => setError('Could not load regulations.'))
  }, [])

  useEffect(() => {
    if (!active) return
    setLoading(true); setError(null)
    regulationChunksAPI(active, 50)
      .then(setChunks)
      .catch(() => setError('Could not load regulation text.'))
      .finally(() => setLoading(false))
  }, [active])

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Regulations</div>
        <div className="page-sub">Browse indexed GDPR · CCPA · HIPAA · NIST text</div>
      </div>

      {error && (
        <div style={{ background: '#3b0e0e', border: '1px solid var(--critical)', borderRadius: 8, padding: 14, marginBottom: 16, color: 'var(--critical)', fontSize: 13 }}>
          {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        {regs.map(r => (
          <button key={r} className={`btn ${r === active ? 'btn-primary' : 'btn-ghost'}`}
            style={{ fontSize: 12 }} onClick={() => setActive(r)}>{r}</button>
        ))}
        {regs.length === 0 && <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>No regulations indexed.</span>}
      </div>

      {loading ? (
        <div className="empty-state"><span className="spinner" /> Loading…</div>
      ) : chunks.length === 0 ? (
        <div className="empty-state"><BookOpen size={32} /> Select a regulation to browse its text.</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {chunks.map((c, i) => (
            <div key={i} className="card">
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, fontFamily: 'var(--font-mono)' }}>
                {c.regulation} · {c.chunk_id || c.paper_id}
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-dim)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{c.content}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
