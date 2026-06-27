// src/pages/Dashboard.tsx
import { useEffect, useState } from 'react'
import { healthReadyAPI, conflictsAPI, Conflict, API_TARGET } from '../lib/api'
import { AlertTriangle, CheckCircle, GitMerge } from 'lucide-react'

export default function Dashboard() {
  const [health, setHealth] = useState<{ status: string; pipeline_ready: boolean; version?: string } | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [conflicts, setConflicts] = useState<Conflict[]>([])

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const poll = () => {
      healthReadyAPI()
        .then(data => {
          if (cancelled) return
          setHealth(data)
          setHealthError(null)
          if (!data.pipeline_ready) timer = setTimeout(poll, 3000)
        })
        .catch((err) => {
          if (cancelled) return
          const msg = err?.message || 'Unknown error'
          console.error('[Dashboard] /health/ready check failed:', msg, err)
          setHealth({ status: 'unreachable', pipeline_ready: false })
          setHealthError(msg)
          timer = setTimeout(poll, 5000)
        })
    }
    poll()

    conflictsAPI().then(setConflicts).catch(() => setConflicts([]))

    return () => { cancelled = true; clearTimeout(timer) }
  }, [])

  const ready = health?.pipeline_ready ?? false

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Privacy Compliance Dashboard</div>
        <div className="page-sub">Automated auditing against GDPR · CCPA · HIPAA · NIST</div>
      </div>

      {/* Status bar */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {ready
            ? <CheckCircle size={16} color="var(--low)" />
            : <AlertTriangle size={16} color="var(--high)" />}
          <span style={{ fontSize: 13 }}>
            Pipeline status: <strong style={{ color: ready ? 'var(--low)' : 'var(--high)' }}>
              {ready ? 'Ready' : health?.status === 'unreachable' ? 'Server unreachable' : 'Loading…'}
            </strong>
          </span>
          {ready && (
            <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              v{(health as any)?.version ?? '—'}
            </span>
          )}
        </div>
        {health?.status === 'unreachable' && (
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', lineHeight: 1.6 }}>
            Proxy target (VITE_API_URL): {API_TARGET}
            {healthError && <><br />Error: {healthError}</>}
            <br />Check that this matches your current ngrok URL, then restart `npm run dev`.
          </div>
        )}
      </div>

      {/* Cross-regulation conflicts */}
      {conflicts.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <GitMerge size={12} /> Cross-Regulation Conflicts ({conflicts.length})
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {conflicts.map((cf, i) => (
              <div key={i} style={{ fontSize: 13, color: 'var(--text-dim)' }}>
                <strong>{cf.source}</strong>
                {cf.value_a ? ` (${cf.value_a})` : ''}{' '}
                <span style={{ color: 'var(--text-muted)' }}>{(cf.rel_type || 'CONFLICTS_WITH').replace(/_/g, ' ').toLowerCase()}</span>{' '}
                <strong>{cf.target}</strong>
                {cf.value_b ? ` (${cf.value_b})` : ''}
                {cf.description ? ` — ${cf.description}` : ''}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* How to use */}
      <div className="two-col">
        <div className="card">
          <div className="card-title">How to run an audit</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {[
              { step: '1', text: 'Go to Audit and upload a sample privacy policy PDF' },
              { step: '2', text: 'Ask a general compliance question, then select a policy from the dropdown' },
              { step: '3', text: 'Review the detected gaps and remediation checklist' },
              { step: '4', text: 'View past audits in History' },
            ].map(({ step, text }) => (
              <div key={step} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <div style={{
                  width: 22, height: 22, borderRadius: '50%',
                  background: 'var(--accent-dim)', color: 'var(--accent)',
                  fontSize: 11, fontWeight: 700, display: 'flex',
                  alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                }}>{step}</div>
                <div style={{ fontSize: 13, color: 'var(--text-dim)', paddingTop: 3 }}>{text}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title">Sample compliance questions</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              'Does the policy comply with GDPR Article 17 right to erasure?',
              'What CCPA rights are missing from this privacy policy?',
              'Does this policy address HIPAA breach notification requirements?',
              'Are there any conflicts between GDPR and NIST requirements?',
            ].map((q, i) => (
              <div key={i} style={{
                background: 'var(--surface-2)', border: '1px solid var(--border)',
                borderRadius: 6, padding: '8px 12px',
                fontSize: 12, color: 'var(--text-dim)', fontStyle: 'italic',
              }}>
                "{q}"
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Regulation reference */}
      <div className="card">
        <div className="card-title">Regulation reference</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          {[
            { reg: 'GDPR', desc: 'General Data Protection Regulation', region: 'EU', penalty: '€20M or 4% turnover', color: 'var(--info)' },
            { reg: 'CCPA', desc: 'California Consumer Privacy Act', region: 'US · California', penalty: '$7,500 per violation', color: 'var(--low)' },
            { reg: 'HIPAA', desc: 'Health Insurance Portability Act', region: 'US · Healthcare', penalty: 'Up to $1.9M per year', color: 'var(--medium)' },
            { reg: 'NIST', desc: 'Cybersecurity Framework', region: 'US · Federal', penalty: 'Contractual penalties', color: 'var(--high)' },
          ].map(({ reg, desc, region, penalty, color }) => (
            <div key={reg} style={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 16, fontWeight: 700, fontFamily: 'var(--font-mono)', color, marginBottom: 4 }}>{reg}</div>
              <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>{desc}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{region}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>Max: {penalty}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
