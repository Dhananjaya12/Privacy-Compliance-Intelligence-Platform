// src/pages/Ingest.tsx
import { useState, useRef } from 'react'
import { ingestAPI, IngestResponse } from '../lib/api'
import { UploadCloud, FileText, X, CheckCircle } from 'lucide-react'

function fmtSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export default function Ingest({ onIngested, compact }: { onIngested?: () => void; compact?: boolean } = {}) {
  const [files, setFiles]       = useState<File[]>([])
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [result, setResult]     = useState<IngestResponse | null>(null)
  const [error, setError]       = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return
    const pdfs = Array.from(incoming).filter(f => f.type === 'application/pdf')
    setFiles(prev => [...prev, ...pdfs])
    setResult(null); setError(null)
  }

  const remove = (i: number) => setFiles(prev => prev.filter((_, idx) => idx !== i))

  const upload = async () => {
    if (!files.length) return
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await ingestAPI(files)
      setResult(res)
      setFiles([])
      onIngested?.()
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Upload failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      {!compact && (
        <div className="page-header">
          <div className="page-title">Ingest Documents</div>
          <div className="page-sub">Upload PDF privacy policies to index them for compliance auditing</div>
        </div>
      )}

      {/* Drop zone */}
      <div
        className={`upload-zone ${dragging ? 'drag-over' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}
      >
        <UploadCloud size={32} className="upload-zone-icon" />
        <div className="upload-zone-text">Drop PDF files here or click to browse</div>
        <div className="upload-zone-sub">Supports multiple files · Max 50 MB each</div>
        <input ref={inputRef} type="file" accept=".pdf" multiple hidden onChange={e => addFiles(e.target.files)} />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="file-list">
          {files.map((f, i) => (
            <div key={i} className="file-item">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <FileText size={14} color="var(--accent)" />
                <span className="file-item-name">{f.name}</span>
                <span className="file-item-size">{fmtSize(f.size)}</span>
              </div>
              <button onClick={() => remove(i)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                <X size={14} />
              </button>
            </div>
          ))}

          <button className="btn btn-primary" onClick={upload} disabled={loading} style={{ marginTop: 8, alignSelf: 'flex-start' }}>
            {loading ? <span className="spinner" /> : <UploadCloud size={14} />}
            {loading ? 'Ingesting…' : `Ingest ${files.length} file${files.length > 1 ? 's' : ''}`}
          </button>
        </div>
      )}

      {/* Success */}
      {result && (
        <div style={{ background: '#0a2e14', border: '1px solid var(--low)', borderRadius: 8, padding: 16, marginTop: 16, display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          <CheckCircle size={16} color="var(--low)" style={{ flexShrink: 0, marginTop: 1 }} />
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--low)', marginBottom: 4 }}>Ingestion complete</div>
            <div style={{ fontSize: 13, color: 'var(--text-dim)' }}>{result.message}</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6, fontFamily: 'var(--font-mono)' }}>
              {result.files_processed} file{result.files_processed !== 1 ? 's' : ''} · {result.chunks_created.toLocaleString()} chunks indexed → policies index
            </div>
            {result.paper_ids?.length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                {result.paper_ids.map(p => <span key={p} className="chip" style={{ fontSize: 10 }}>{p}</span>)}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ background: '#3b0e0e', border: '1px solid var(--critical)', borderRadius: 8, padding: 14, marginTop: 16, color: 'var(--critical)', fontSize: 13 }}>
          {error}
        </div>
      )}

      {/* Tips */}
      <div className="card" style={{ marginTop: 24 }}>
        <div className="card-title">Tips</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {[
            'Text-based PDFs are extracted instantly with pymupdf4llm. Scanned PDFs fall back to Azure OCR.',
            'For bulk ingestion of many files, use spark_ingestion.py directly — it parallelises across CPU cores.',
            'Already-indexed documents are detected by filename — re-uploading the same PDF skips re-processing.',
            'After ingestion, go to Audit and ask a compliance question to see the gap analysis.',
          ].map((tip, i) => (
            <div key={i} style={{ fontSize: 13, color: 'var(--text-dim)', paddingLeft: 12, borderLeft: '2px solid var(--border)' }}>
              {tip}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
