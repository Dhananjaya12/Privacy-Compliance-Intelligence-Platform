// src/lib/api.ts
import axios from 'axios'

// ngrok-skip-browser-warning avoids ngrok's free-tier HTML interstitial page,
// which would otherwise be returned instead of JSON for proxied requests.
const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'ngrok-skip-browser-warning': 'true' },
})

// The backend URL the Vite dev server proxies /api/* to (set in frontend/.env
// as VITE_API_URL — must match the *current* ngrok URL, which changes every
// time the Colab tunnel restarts).
export const API_TARGET = import.meta.env.VITE_API_URL || '(not set — see vite.config.ts fallback)'
console.info(`[api] /api/* is proxied to: ${API_TARGET}`)

// Log full details for any failed request so network/CORS/ngrok issues are
// easy to diagnose from the browser console.
api.interceptors.response.use(
  (res) => res,
  (err) => {
    console.error('[api] request failed', {
      url:     `${err.config?.baseURL ?? ''}${err.config?.url ?? ''}`,
      method:  err.config?.method,
      status:  err.response?.status,
      message: err.message,
      data:    err.response?.data,
    })
    return Promise.reject(err)
  }
)

// ── Types ─────────────────────────────────────────────────────────────────────


const KNOWN_POLICY_ALIASES: Record<string, string> = {
  'microsoft_privacy_policy.pdf': 'privacy_policy_1.pdf',
  'meta_privacy_policy.pdf': 'privacy_policy_2.pdf',
  'stripe_privacy_policy.pdf': 'privacy_policy_3.pdf',
  'uber_privacy_policy.pdf': 'privacy_policy_4.pdf',
  'salesforce_privacy_policy.pdf': 'privacy_policy_5.pdf',
  'netflix_privacy_policy.pdf': 'privacy_policy_6.pdf',
  'google_privacy_policy_latest.pdf': 'privacy_policy_7.pdf',
  'zoom_privacy_policy.pdf': 'privacy_policy_8.pdf',
}

const COMPANY_NAME_REPLACEMENTS: [RegExp, string][] = [
  [/microsoft/gi, 'Policy 1'],
  [/meta/gi, 'Policy 2'],
  [/stripe/gi, 'Policy 3'],
  [/uber/gi, 'Policy 4'],
  [/salesforce/gi, 'Policy 5'],
  [/netflix/gi, 'Policy 6'],
  [/google/gi, 'Policy 7'],
  [/zoom/gi, 'Policy 8'],
]

function fallbackPolicyAlias(filename: string): string {
  let hash = 0
  for (let i = 0; i < filename.length; i += 1) {
    hash = ((hash << 5) - hash + filename.charCodeAt(i)) | 0
  }
  return `privacy_policy_${Math.abs(hash % 900) + 100}.pdf`
}

export const anonymizePolicyName = (filename?: string | null): string => {
  if (!filename) return 'privacy_policy.pdf'
  const clean = filename.split(/[\\/]/).pop() || filename
  const key = clean.toLowerCase()
  if (KNOWN_POLICY_ALIASES[key]) return KNOWN_POLICY_ALIASES[key]
  if (key.includes('privacy') || key.endsWith('.pdf')) return fallbackPolicyAlias(clean)
  return clean
}

export const anonymizeText = (value?: string | null): string => {
  if (!value) return ''
  let output = value
  Object.entries(KNOWN_POLICY_ALIASES).forEach(([realName, alias]) => {
    const stem = realName.replace(/\.pdf$/i, '')
    const looseStem = stem.replace(/[_-]/g, '[ _-]')
    output = output.replace(new RegExp(realName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), alias)
    output = output.replace(new RegExp(looseStem, 'gi'), alias.replace(/\.pdf$/i, ''))
  })
  COMPANY_NAME_REPLACEMENTS.forEach(([pattern, alias]) => {
    output = output.replace(pattern, alias)
  })
  return output
}

export const anonymizePolicyList = (filenames: string[]): { real: string; alias: string }[] =>
  filenames.map(real => ({ real, alias: anonymizePolicyName(real) }))

export interface SourceChunk {
  paper_id: string
  page: number
  text: string
}

export interface Gap {
  obligation_id: string
  regulation: string
  description: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  article: string
  ob_type: string
  document?: string
}

export interface Conflict {
  source: string
  target: string
  rel_type?: string
  description?: string
  concept?: string
  value_a?: string
  value_b?: string
  unit?: string
}

export interface Remediation {
  obligation_id: string
  regulation: string
  document?: string
  severity?: string
  recommendation: string
}

export interface GapGroupRemediation {
  theme: string
  label: string
  regulations: string[]
  severity: string
  recommendation: string
  document?: string
}

export interface GapGroup {
  theme: string
  label: string
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info'
  regulations: string[]
  gaps: Gap[]
  remediation?: GapGroupRemediation
}

export interface ComplianceDetail {
  jurisdictions: string[]
  documents: string[]
  compliance_score: number | null      // 0-100, higher = better
  per_reg_compliance: Record<string, number>
  overall_risk: number | null
  gaps: Gap[]
  conflicts: Conflict[]
  remediations: Remediation[]
  financial_exposure: string
  gap_groups: GapGroup[]
  obligation_counts: Record<string, number>
}

export interface QueryResponse {
  query: string
  answer: string
  source_chunks: SourceChunk[]
  clarification: string | null
  query_intent: 'audit' | 'coverage'
  compliance: ComplianceDetail | null
}

export interface IngestResponse {
  message: string
  files_processed: number
  chunks_created: number
  paper_ids: string[]
}

export interface HealthResponse {
  status: string
  pipeline_ready: boolean
  version: string
}

export interface HistoryRun {
  run_id: string
  policy_name: string | null
  query: string | null
  compliance_score: number | null
  overall_risk: number | null
  total_gaps: number | null
  total_conflicts: number | null
  jurisdictions: string | null
  start_time: string
  answer?: string
  query_intent?: 'audit' | 'coverage'
  compliance?: ComplianceDetail | null
}


const LOCAL_HISTORY_KEY = 'complianceAuditHistory'

export const loadLocalHistory = (): HistoryRun[] => {
  try {
    const raw = window.localStorage.getItem(LOCAL_HISTORY_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export const saveLocalHistory = (
  question: string,
  response: QueryResponse,
  selectedPolicy?: string,
): void => {
  const compliance = response.compliance
  const item: HistoryRun = {
    run_id: `local-${Date.now()}`,
    policy_name: selectedPolicy || compliance?.documents?.[0] || null,
    query: response.query || question,
    compliance_score: compliance?.compliance_score ?? null,
    overall_risk: compliance?.overall_risk ?? null,
    total_gaps: compliance?.gaps?.length ?? null,
    total_conflicts: compliance?.conflicts?.length ?? null,
    jurisdictions: compliance?.jurisdictions?.join(', ') || null,
    start_time: String(Date.now()),
    answer: response.answer,
    query_intent: response.query_intent,
    compliance,
  }

  const existing = loadLocalHistory()
  const next = [item, ...existing].slice(0, 100)
  window.localStorage.setItem(LOCAL_HISTORY_KEY, JSON.stringify(next))
}

export const clearLocalHistory = (): void => {
  window.localStorage.removeItem(LOCAL_HISTORY_KEY)
}

export interface TrendPoint {
  start_time: string
  policy_name: string | null
  compliance_score: number | null
  total_gaps: number | null
}

export interface GraphData {
  nodes: { id: string; label: string }[]
  links: { source: string; target: string; type: string; description?: string; value_a?: string; value_b?: string }[]
}

export interface RegulationChunk {
  regulation: string
  paper_id: string
  chunk_id: string
  content: string
}

// ── Streaming query progress ────────────────────────────────────────────────

export interface QueryProgressEvent {
  type: 'progress'
  node: string
  label: string
}

export interface QueryDoneEvent {
  type: 'done'
  result: QueryResponse
}

export interface QueryErrorEvent {
  type: 'error'
  message: string
}

export type QueryStreamEvent = QueryProgressEvent | QueryDoneEvent | QueryErrorEvent

// ── API calls ─────────────────────────────────────────────────────────────────

export const queryAPI = async (question: string, policyDocument?: string): Promise<QueryResponse> => {
  const { data } = await api.post<QueryResponse>('/query', {
    query: question,
    policy_document: policyDocument ?? null,
  })
  return data
}

// Streams /query/stream (SSE). Calls `onProgress` for each pipeline stage as
// it completes, then resolves with the same payload `queryAPI` would return.
export const queryStreamAPI = async (
  question: string,
  policyDocument: string | undefined,
  onProgress: (event: QueryProgressEvent) => void,
): Promise<QueryResponse> => {
  const res = await fetch('/api/v1/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
    body: JSON.stringify({ query: question, policy_document: policyDocument ?? null }),
  })

  if (!res.ok || !res.body) {
    throw new Error(`Query stream failed: ${res.status} ${res.statusText}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const line = frame.trim()
      if (!line.startsWith('data:')) continue

      const event = JSON.parse(line.slice(5).trim()) as QueryStreamEvent
      if (event.type === 'progress') {
        onProgress(event)
      } else if (event.type === 'done') {
        return event.result
      } else if (event.type === 'error') {
        throw new Error(event.message)
      }
    }
  }

  throw new Error('Query stream ended without a result')
}

export const ingestAPI = async (files: File[]): Promise<IngestResponse> => {
  const form = new FormData()
  files.forEach(f => form.append('files', f))
  const { data } = await api.post<IngestResponse>('/ingest', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const healthAPI = async (): Promise<HealthResponse> => {
  const { data } = await api.get<HealthResponse>('/health')
  return data
}

// /health is a plain liveness probe (no pipeline_ready field); /health/ready
// reports whether the agent/retriever/pipeline have finished loading.
export const healthReadyAPI = async (): Promise<HealthResponse> => {
  const { data } = await api.get<HealthResponse>('/health/ready')
  return data
}

export const listPoliciesAPI = async (): Promise<string[]> => {
  const { data } = await api.get<{ policies: string[] }>('/policies')
  return data.policies ?? []
}

export const historyAPI = async (limit = 50): Promise<HistoryRun[]> => {
  const { data } = await api.get<{ runs: HistoryRun[] }>('/history', { params: { limit } })
  return data.runs ?? []
}

export const trendsAPI = async (policyName?: string): Promise<TrendPoint[]> => {
  const { data } = await api.get<{ points: TrendPoint[] }>('/trends', {
    params: policyName ? { policy_name: policyName } : {},
  })
  return data.points ?? []
}

export const conflictsAPI = async (limit = 50): Promise<Conflict[]> => {
  const { data } = await api.get<{ conflicts: Conflict[] }>('/conflicts', { params: { limit } })
  return data.conflicts ?? []
}

export const graphAPI = async (limit = 200, regulation?: string): Promise<GraphData> => {
  const { data } = await api.get<GraphData>('/graph', {
    params: { limit, ...(regulation ? { regulation } : {}) },
  })
  return data
}

export const regulationsAPI = async (): Promise<string[]> => {
  const { data } = await api.get<{ regulations: string[] }>('/regulations')
  return data.regulations ?? []
}

export const regulationChunksAPI = async (regulation?: string, top = 50): Promise<RegulationChunk[]> => {
  const { data } = await api.get<{ chunks: RegulationChunk[] }>('/regulations/chunks', {
    params: { top, ...(regulation ? { regulation } : {}) },
  })
  return data.chunks ?? []
}

export const downloadReportPDF = async (markdown: string, filename = 'compliance_report.pdf'): Promise<void> => {
  let res
  try {
    res = await api.post('/report/pdf', { markdown, filename }, { responseType: 'blob' })
  } catch (err: any) {
    const blob = err?.response?.data
    if (blob instanceof Blob && blob.type.includes('json')) {
      const text = await blob.text()
      let detail: string | undefined
      try { detail = JSON.parse(text)?.detail } catch { /* not JSON */ }
      throw new Error(detail || 'PDF generation failed.')
    }
    throw new Error(err?.message || 'PDF generation failed.')
  }
  const url = window.URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  window.URL.revokeObjectURL(url)
}
