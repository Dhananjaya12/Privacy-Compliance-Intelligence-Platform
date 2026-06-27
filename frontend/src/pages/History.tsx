// src/pages/History.tsx
import { useState, useEffect } from 'react'
import { anonymizePolicyName, anonymizeText, clearLocalHistory, historyAPI, HistoryRun, loadLocalHistory } from '../lib/api'
import { Clock, Download, RefreshCw, Trash2 } from 'lucide-react'

function escapeHtml(value: unknown): string {
  return String(value ?? '?')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

function severityColor(severity: string): string {
  if (severity === 'critical') return '#dc2626'
  if (severity === 'high') return '#ea580c'
  if (severity === 'medium') return '#d97706'
  if (severity === 'low') return '#16a34a'
  return '#6b7280'
}

function frameworkLabel(run: HistoryRun): string {
  const values = new Set<string>()
  run.compliance?.jurisdictions?.forEach(j => j && values.add(j))
  run.compliance?.gaps?.forEach(g => g.regulation && values.add(g.regulation))
  ;(run.jurisdictions || '').split(',').map(v => v.trim()).filter(Boolean).forEach(v => values.add(v))
  return Array.from(values).join(', ') || '?'
}

function summarizeAudit(run: HistoryRun): string {
  const c = run.compliance
  if (!c) {
    return `Audited ${escapeHtml(anonymizePolicyName(run.policy_name))}. Gaps found: ${escapeHtml(run.total_gaps ?? '?')}.`
  }

  const docs = c.documents?.length ? c.documents.map(anonymizePolicyName).join(', ') : anonymizePolicyName(run.policy_name) || 'the selected policy'
  const frameworks = frameworkLabel(run) || 'detected compliance frameworks'
  const critical = c.gaps?.filter(g => g.severity === 'critical').length ?? 0
  const high = c.gaps?.filter(g => g.severity === 'high').length ?? 0
  const medium = c.gaps?.filter(g => g.severity === 'medium').length ?? 0
  const low = c.gaps?.filter(g => g.severity === 'low').length ?? 0
  const parts = [
    critical ? `<strong style="color:#dc2626">${critical} critical</strong>` : '',
    high ? `<strong style="color:#ea580c">${high} high</strong>` : '',
    medium ? `<strong style="color:#d97706">${medium} medium</strong>` : '',
    low ? `<strong style="color:#16a34a">${low} low</strong>` : '',
  ].filter(Boolean).join(', ')

  if ((c.gaps?.length ?? 0) === 0) {
    return `Audited <strong>${escapeHtml(docs)}</strong> against <strong>${escapeHtml(frameworks)}</strong>. No compliance gaps were identified for this query.`
  }

  return `Audited <strong>${escapeHtml(docs)}</strong> against <strong>${escapeHtml(frameworks)}</strong>. Found <strong>${c.gaps.length}</strong> gap${c.gaps.length === 1 ? '' : 's'} - ${parts}.`
}

function buildGapsHtml(run: HistoryRun): string {
  const gaps = run.compliance?.gaps ?? []
  if (!gaps.length) return '<p class="muted">No detailed gaps were recorded for this audit.</p>'

  return `<table>
    <thead><tr><th>Severity</th><th>Regulation</th><th>Type</th><th>Description</th><th>Article</th></tr></thead>
    <tbody>
      ${gaps.map(g => `<tr>
        <td><span class="badge" style="background:${severityColor(g.severity)}">${escapeHtml(g.severity)}</span></td>
        <td>${escapeHtml(g.regulation)}</td>
        <td>${escapeHtml(g.ob_type)}</td>
        <td>${escapeHtml(anonymizeText(g.description))}</td>
        <td>${escapeHtml(g.article)}</td>
      </tr>`).join('')}
    </tbody>
  </table>`
}

function buildRemediationHtml(run: HistoryRun): string {
  const groups = run.compliance?.gap_groups ?? []
  const remediations = run.compliance?.remediations ?? []

  if (groups.length) {
    return `<div class="checklist">
      ${groups.map(group => `<div class="check-item">
        <div><span class="badge" style="background:${severityColor(group.severity)}">${escapeHtml(group.severity)}</span> <strong>${escapeHtml(anonymizeText(group.label))}</strong></div>
        <div class="muted">Frameworks: ${escapeHtml(group.regulations?.join(', '))} - Findings: ${group.gaps?.length ?? 0}</div>
        ${group.remediation?.recommendation ? `<p>${escapeHtml(anonymizeText(group.remediation.recommendation))}</p>` : ''}
      </div>`).join('')}
    </div>`
  }

  if (remediations.length) {
    return `<div class="checklist">
      ${remediations.map(r => `<div class="check-item">
        <div><span class="badge" style="background:${severityColor(r.severity || 'info')}">${escapeHtml(r.severity || 'info')}</span> <strong>${escapeHtml(r.regulation)}</strong></div>
        <p>${escapeHtml(anonymizeText(r.recommendation))}</p>
      </div>`).join('')}
    </div>`
  }

  return '<p class="muted">No remediation checklist was recorded for this audit.</p>'
}

function buildHistoryReportHtml(runs: HistoryRun[]): string {
  const generatedAt = new Date().toLocaleString()
  const sections = runs.map((r, i) => {
    const when = r.start_time ? new Date(Number(r.start_time)).toLocaleString() : '?'
    return `<section class="audit-card">
      <div class="audit-kicker">Audit ${i + 1} - ${escapeHtml(when)}</div>
      <h2>${escapeHtml(anonymizeText(r.query || 'Compliance audit'))}</h2>
      <div class="chips">
        <span>Document: ${escapeHtml(anonymizePolicyName(r.policy_name))}</span>
        <span>Frameworks: ${escapeHtml(frameworkLabel(r))}</span>
      </div>
      <h3>Audit Summary</h3>
      <p>${summarizeAudit(r)}</p>
      <h3>Identified Gaps</h3>
      ${buildGapsHtml(r)}
      <h3>Remediation Checklist</h3>
      ${buildRemediationHtml(r)}
    </section>`
  }).join('')

  return `<!doctype html>
  <html>
    <head>
      <title>Compliance Audit Report</title>
      <style>
        body { font-family: Arial, sans-serif; color: #111827; margin: 32px; line-height: 1.5; background: #f9fafb; }
        .toolbar { position: sticky; top: 0; background: #f9fafb; padding: 0 0 16px; text-align: right; }
        button { padding: 9px 14px; border: 1px solid #d1d5db; border-radius: 8px; background: white; cursor: pointer; }
        h1 { margin-bottom: 4px; }
        h2 { margin: 4px 0 10px; font-size: 22px; }
        h3 { margin-top: 22px; margin-bottom: 8px; font-size: 14px; text-transform: uppercase; letter-spacing: .06em; color: #4b5563; }
        .meta, .muted { color: #6b7280; }
        .audit-card { background: white; border: 1px solid #d1d5db; border-radius: 14px; padding: 22px; margin: 22px 0; page-break-inside: avoid; }
        .audit-kicker { color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
        .chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 18px; }
        .chips span { border: 1px solid #d1d5db; border-radius: 999px; padding: 5px 9px; font-size: 12px; color: #374151; background: #f9fafb; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th, td { border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }
        th { background: #f3f4f6; text-align: left; }
        .badge { display: inline-block; color: white; border-radius: 999px; padding: 3px 8px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
        .checklist { display: grid; gap: 10px; }
        .check-item { border-left: 4px solid #2563eb; background: #f9fafb; padding: 10px 12px; border-radius: 8px; }
        @media print { .toolbar { display: none; } body { background: white; margin: 16mm; } .audit-card { border-color: #9ca3af; } }
      </style>
    </head>
    <body>
      <div class="toolbar"><button onclick="window.print()">Save as PDF</button></div>
      <h1>Compliance Audit Report</h1>
      <div class="meta">Generated ${escapeHtml(generatedAt)} - ${runs.length} audit${runs.length === 1 ? '' : 's'} included</div>
      ${sections || '<section class="audit-card">No audits recorded.</section>'}
    </body>
  </html>`
}

export default function History() {
  const [runs, setRuns]       = useState<HistoryRun[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)

  const load = () => {
    setLoading(true); setError(null)
    historyAPI(100)
      .then((serverRuns) => {
        const localRuns = loadLocalHistory()
        const seen = new Set<string>()
        const merged = [...localRuns, ...serverRuns].filter((run) => {
          const key = run.run_id || `${run.start_time}-${run.query}`
          if (seen.has(key)) return false
          seen.add(key)
          return true
        })
        setRuns(merged)
      })
      .catch(() => {
        setRuns(loadLocalHistory())
        setError('Could not load MLflow history, so showing browser-saved demo history instead.')
      })
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const generateReport = () => {
    const html = buildHistoryReportHtml(runs)
    const blob = new Blob([html], { type: 'text/html;charset=utf-8' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `compliance-audit-history-report-${new Date().toISOString().slice(0, 10)}.html`
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
  }

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div className="page-title">Audit History</div>
          <div className="page-sub">Recent compliance audits saved locally for the demo, with MLflow history when available</div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <button className="btn btn-primary" onClick={generateReport} disabled={runs.length === 0} style={{ fontSize: 12 }}>
            <Download size={12} /> Download audit reports
          </button>
          <button className="btn btn-ghost" onClick={load} style={{ fontSize: 12 }}>
            <RefreshCw size={12} /> Refresh
          </button>
          <button
            className="btn btn-ghost"
            onClick={() => { clearLocalHistory(); load() }}
            style={{ fontSize: 12 }}
          >
            <Trash2 size={12} /> Clear demo history
          </button>
        </div>
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
              <tr><th>When</th><th>Query</th><th>Gaps</th><th>Conflicts</th><th>Frameworks</th></tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                    {r.start_time ? new Date(Number(r.start_time)).toLocaleString() : '?'}
                  </td>
                  <td style={{ maxWidth: 460, fontSize: 12, color: 'var(--text-dim)' }}>{anonymizeText(r.query) || '?'}</td>
                  <td>{r.total_gaps ?? '?'}</td>
                  <td>{r.total_conflicts ?? '?'}</td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{frameworkLabel(r)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
