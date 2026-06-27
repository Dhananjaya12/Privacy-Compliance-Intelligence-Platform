// src/pages/Audit.tsx
import { useEffect, useState } from 'react'
import {
  queryStreamAPI, listPoliciesAPI,
  QueryResponse, QueryProgressEvent, Gap, GapGroup, saveLocalHistory, anonymizePolicyName, anonymizeText,
} from '../lib/api'
import Ingest from './Ingest'
import { Search, AlertTriangle, Lightbulb, CheckCircle, Circle, UploadCloud, ChevronDown, ChevronUp } from 'lucide-react'

const SEVERITY_ORDER = ['critical', 'high', 'medium', 'low'] as const
type Severity = typeof SEVERITY_ORDER[number]

// Mirrors agent/graph.py's node order + app/services/rag_service.py's NODE_LABELS,
// shown as a progress checklist while a streamed audit runs.
const PIPELINE_STEPS: { node: string; label: string }[] = [
  { node: 'doc_resolver',          label: 'Identifying target document(s)' },
  { node: 'jurisdiction_detector', label: 'Detecting applicable regulations' },
  { node: 'kg_retriever',          label: 'Retrieving obligations from knowledge graph' },
  { node: 'gap_analyzer',          label: 'Analyzing compliance gaps' },
  { node: 'conflict_detector',     label: 'Checking cross-regulation conflicts' },
  { node: 'risk_scorer',           label: 'Assessing risk severity' },
  { node: 'remediation',           label: 'Generating remediation recommendations' },
  { node: 'report_generator',      label: 'Generating report' },
]


const SAMPLE_QUESTIONS = [
  'Does this selected policy comply with GDPR Article 17 right to erasure?',
  'What CCPA rights are missing from this privacy policy?',
  'Which policies address HIPAA breach notification timelines?',
  'Are data retention schedules documented per NIST requirements?',
]

export default function Audit() {
  const [query, setQuery]       = useState('')
  const [policies, setPolicies] = useState<string[]>([])
  const [target, setTarget]     = useState<string>('')      // '' = auto-detect
  const [loading, setLoading]   = useState(false)
  const [result, setResult]     = useState<QueryResponse | null>(null)
  const [error, setError]       = useState<string | null>(null)
  const [resolvedThemes, setResolvedThemes] = useState<Set<string>>(new Set())
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set())
  const [showIngest, setShowIngest] = useState(false)

  useEffect(() => {
    listPoliciesAPI().then(setPolicies).catch(() => setPolicies([]))
  }, [])

  const toggleTheme = (theme: string) => {
    setResolvedThemes(prev => {
      const next = new Set(prev)
      if (next.has(theme)) next.delete(theme)
      else next.add(theme)
      return next
    })
  }

  const run = async (q: string) => {
    if (!q.trim()) return
    setLoading(true); setError(null); setResult(null); setResolvedThemes(new Set()); setCompletedSteps(new Set())
    try {
      const res = await queryStreamAPI(q, target || undefined, (event: QueryProgressEvent) => {
        setCompletedSteps(prev => new Set(prev).add(event.node))
      })
      setResult(res)
      saveLocalHistory(q, res, target || undefined)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Request failed. Is the server running?')
    } finally {
      setLoading(false)
    }
  }

  const c = result?.compliance

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Compliance Audit</div>
        <div className="page-sub">Ask a question about any ingested privacy policy</div>
      </div>

      {/* Ingest documents (collapsible) */}
      <div className="card" style={{ marginBottom: 20 }}>
        <button
          onClick={() => setShowIngest(v => !v)}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            width: '100%', background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--text)', padding: 0, font: 'inherit',
          }}
        >
          <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6, margin: 0 }}>
            <UploadCloud size={14} /> Ingest documents
          </span>
          {showIngest ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {showIngest && (
          <div style={{ marginTop: 16 }}>
            <Ingest compact onIngested={() => listPoliciesAPI().then(setPolicies).catch(() => {})} />
          </div>
        )}
      </div>

      {/* Query bar + target selector */}
      <div className="query-bar">
        <input
          className="query-input"
          placeholder="e.g. Does this selected policy comply with GDPR Article 17?"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && run(query)}
        />
        <select className="btn btn-ghost" value={target} onChange={e => setTarget(e.target.value)}
          style={{ maxWidth: 220 }}>
          <option value="">Auto-detect document</option>
          {policies.map(p => <option key={p} value={p}>{anonymizePolicyName(p)}</option>)}
        </select>
        <button className="btn btn-primary" onClick={() => run(query)} disabled={loading || !query.trim()}>
          {loading ? <span className="spinner" /> : <Search size={14} />}
          {loading ? 'Analysing…' : 'Run audit'}
        </button>
      </div>

      {/* Sample questions */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
        {SAMPLE_QUESTIONS.map((q, i) => (
          <button key={i} className="btn btn-ghost" style={{ fontSize: 11 }}
            onClick={() => { setQuery(q); run(q) }}>
            {q.slice(0, 48)}…
          </button>
        ))}
      </div>

      {error && (
        <div style={{ background: '#3b0e0e', border: '1px solid var(--critical)', borderRadius: 8, padding: 14, marginBottom: 16, color: 'var(--critical)', fontSize: 13 }}>
          <AlertTriangle size={14} style={{ display: 'inline', marginRight: 8 }} />
          {error}
        </div>
      )}

      {/* Progress checklist while the streamed audit runs */}
      {loading && (
        <div className="card">
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="spinner" /> Running compliance audit…
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {PIPELINE_STEPS.map((step, i) => {
              const done = completedSteps.has(step.node)
              const isCurrent = !done && PIPELINE_STEPS.slice(0, i).every(s => completedSteps.has(s.node))
              return (
                <div key={step.node} style={{
                  display: 'flex', alignItems: 'center', gap: 8, fontSize: 13,
                  color: done ? 'var(--text-dim)' : isCurrent ? 'var(--text)' : 'var(--text-muted)',
                }}>
                  {done
                    ? <CheckCircle size={14} style={{ color: 'var(--low)', flexShrink: 0 }} />
                    : isCurrent
                      ? <span className="spinner" style={{ flexShrink: 0 }} />
                      : <Circle size={14} style={{ flexShrink: 0 }} />}
                  {step.label}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Clarification (ambiguous query) */}
      {result?.clarification && (
        <div className="card" style={{ borderColor: 'var(--medium)' }}>
          <div className="card-title">Clarification needed</div>
          <div style={{ fontSize: 13, color: 'var(--text-dim)' }}>{anonymizeText(result.clarification)}</div>
        </div>
      )}

      {result && c && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Audit summary (plain-English, not a "report" dump) */}
          <div className="card">
            <div className="card-title">
              {result.query_intent === 'coverage' ? 'Coverage Summary' : 'Audit Summary'}
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-dim)', lineHeight: 1.7 }}>
              {result.query_intent === 'coverage' ? (
                <>
                  Searched <strong>{c.documents.length}</strong> ingested{' '}
                  {c.documents.length === 1 ? 'policy' : 'policies'} for content related to this topic
                  {c.jurisdictions.length > 0 && <> under <strong>{c.jurisdictions.join(', ')}</strong></>}.
                  {' '}{c.documents.filter(doc => c.gaps.some(g => g.document === doc)).length > 0
                    ? <><strong style={{ color: 'var(--high)' }}>
                        {c.documents.filter(doc => c.gaps.some(g => g.document === doc)).length}
                      </strong> {c.documents.filter(doc => c.gaps.some(g => g.document === doc)).length === 1 ? 'policy has' : 'policies have'} coverage gaps.</>
                    : <strong style={{ color: 'var(--low)' }}>All policies address this topic adequately.</strong>
                  }
                </>
              ) : (
                <>
                  Audited <strong>{c.documents.map(anonymizePolicyName).join(', ') || 'the policy'}</strong> against{' '}
                  <strong>{c.jurisdictions.join(', ') || 'no detected frameworks'}</strong>.
                  {c.gaps.length > 0 ? (
                    <>
                      {' '}Found <strong>{c.gaps.length}</strong> gap{c.gaps.length === 1 ? '' : 's'}
                      {c.gaps.filter(g => g.severity === 'critical').length > 0 && (
                        <> — <strong style={{ color: 'var(--critical)' }}>
                          {c.gaps.filter(g => g.severity === 'critical').length} critical
                        </strong></>
                      )}
                      {c.gaps.filter(g => g.severity === 'high').length > 0 && (
                        <>, <strong style={{ color: 'var(--high)' }}>
                          {c.gaps.filter(g => g.severity === 'high').length} high
                        </strong></>
                      )}
                      .
                    </>
                  ) : ' No compliance gaps were identified for this query.'}
                  {c.gap_groups.length > 0 && ' See the remediation checklist below for recommended next steps.'}
                </>
              )}
            </div>
            {/* Per-policy breakdown for cross-doc queries */}
            {c.documents.length > 1 && (() => {
              const byDoc = c.documents.map(doc => {
                const docGaps = c.gaps.filter(g => g.document === doc)
                return { doc, gaps: docGaps }
              })
              return (
                <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {byDoc.map(({ doc, gaps: dg }) => {
                    const hasCritical = dg.some(g => g.severity === 'critical')
                    const hasHigh     = dg.some(g => g.severity === 'high')
                    const color = dg.length === 0 ? 'var(--low)' : hasCritical ? 'var(--critical)' : hasHigh ? 'var(--high)' : 'var(--medium)'
                    const label = dg.length === 0 ? 'No gaps found' : `${dg.length} gap${dg.length === 1 ? '' : 's'} — ${hasCritical ? 'critical issues' : hasHigh ? 'high issues' : 'medium/low issues'}`
                    return (
                      <div key={doc} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                        <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                        <strong style={{ color: 'var(--text)' }}>{anonymizePolicyName(doc).replace(/\.[^.]+$/, '')}</strong>
                        <span style={{ color }}>{label}</span>
                      </div>
                    )
                  })}
                </div>
              )
            })()}
          </div>

          {/* Gap Breakdown */}
          <div className="card">
            <div className="card-title">Gap Breakdown</div>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
              {SEVERITY_ORDER.map(sev => {
                const count = c.gaps.filter(g => g.severity === sev).length
                return (
                  <div key={sev} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className={`badge badge-${sev}`}>{sev}</span>
                    <span style={{ fontSize: 22, fontWeight: 700, color: count > 0 ? `var(--${sev})` : 'var(--text-muted)' }}>{count}</span>
                  </div>
                )
              })}
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                {c.jurisdictions.map(j => <span key={j} className="chip">{j}</span>)}
              </div>
            </div>
          </div>

          {/* Gaps table */}
          {c.gaps.length > 0 && (() => {
            const sortedGaps = [...c.gaps].sort((a, b) =>
              SEVERITY_ORDER.indexOf(a.severity as Severity) - SEVERITY_ORDER.indexOf(b.severity as Severity))
            const uniqueDocs = [...new Set(sortedGaps.map(g => g.document).filter(Boolean))]
            const multiDoc = uniqueDocs.length > 1
            const docLabel = (doc?: string) => doc
              ? anonymizePolicyName(doc).replace(/\.[^.]+$/, '').replace(/_/g, ' ').slice(0, 30)
              : '—'
            return (
              <div className="card">
                <div className="card-title">Identified Gaps ({c.gaps.length})</div>
                {multiDoc && (
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {uniqueDocs.map(d => {
                      const gapsForDoc = sortedGaps.filter(g => g.document === d)
                      const hasCritical = gapsForDoc.some(g => g.severity === 'critical')
                      const hasHigh     = gapsForDoc.some(g => g.severity === 'high')
                      const color = hasCritical ? 'var(--critical)' : hasHigh ? 'var(--high)' : 'var(--medium)'
                      return (
                        <span key={d} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                          <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'inline-block' }} />
                          <strong style={{ color: 'var(--text)' }}>{docLabel(d)}</strong>
                          <span style={{ color: 'var(--text-muted)' }}>({gapsForDoc.length} gap{gapsForDoc.length === 1 ? '' : 's'})</span>
                        </span>
                      )
                    })}
                  </div>
                )}
                <table className="gaps-table">
                  <thead>
                    <tr>
                      {multiDoc && <th>Policy</th>}
                      <th>Severity</th><th>Regulation</th><th>Type</th><th>Description</th><th>Article</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedGaps.map((gap: Gap, i) => (
                      <tr key={i}>
                        {multiDoc && (
                          <td style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', maxWidth: 120, wordBreak: 'break-word' }}>
                            {docLabel(gap.document)}
                          </td>
                        )}
                        <td><span className={`badge badge-${gap.severity}`}>{gap.severity}</span></td>
                        <td><span className="chip">{gap.regulation}</span></td>
                        <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>{gap.ob_type}</td>
                        <td style={{ maxWidth: 400 }}>{anonymizeText(gap.description)}</td>
                        <td style={{ color: 'var(--text-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>{gap.article || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          })()}

          {/* Remediation checklist */}
          {c.gap_groups.length > 0 && (
            <div className="card">
              <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Lightbulb size={12} /> Remediation Checklist ({c.gap_groups.length})
                </span>
                {resolvedThemes.size > 0 && (
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {resolvedThemes.size} of {c.gap_groups.length} addressed
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {c.gap_groups.map((group: GapGroup) => {
                  const checked = resolvedThemes.has(group.theme)
                  return (
                    <label key={group.theme}
                      style={{ display: 'flex', gap: 10, alignItems: 'flex-start', cursor: 'pointer', opacity: checked ? 0.5 : 1 }}>
                      <input type="checkbox" checked={checked} onChange={() => toggleTheme(group.theme)}
                        style={{ marginTop: 3 }} />
                      <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                          <span className={`badge badge-${group.severity}`}>{group.severity}</span>
                          <strong style={{ fontSize: 13, textDecoration: checked ? 'line-through' : 'none' }}>
                            {group.label}
                          </strong>
                          {group.regulations.map(r => <span key={r} className="chip" style={{ fontSize: 10 }}>{r}</span>)}
                          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                            {group.gaps.length} finding{group.gaps.length === 1 ? '' : 's'}
                          </span>
                        </div>
                        {group.remediation?.recommendation && (
                          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>
                            {anonymizeText(group.remediation.recommendation)}
                          </div>
                        )}
                      </div>
                    </label>
                  )
                })}
              </div>

            </div>
          )}
        </div>
      )}

      {!result && !loading && !error && (
        <div className="empty-state">
          <Search size={32} />
          Enter a compliance question above to run an audit
        </div>
      )}
    </div>
  )
}
