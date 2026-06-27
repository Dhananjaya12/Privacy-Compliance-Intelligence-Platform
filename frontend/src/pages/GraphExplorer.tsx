// src/pages/GraphExplorer.tsx
import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { graphAPI, GraphData } from '../lib/api'
import { GitBranch, RefreshCw } from 'lucide-react'

interface D3Node extends d3.SimulationNodeDatum { id: string; label: string; displayLabel: string; category: NodeCategory }
interface D3Link extends d3.SimulationLinkDatum<D3Node> { type: string; description?: string }

type NodeCategory = 'regulation' | 'requirement' | 'policy' | 'evidence' | 'risk' | 'recommendation'

const CATEGORY_META: Record<NodeCategory, { label: string; color: string; radius: number }> = {
  regulation: { label: 'Regulation / Framework', color: '#a78bfa', radius: 10 },
  requirement: { label: 'Compliance Requirement', color: '#facc15', radius: 9 },
  policy: { label: 'Policy Document', color: '#38bdf8', radius: 9 },
  evidence: { label: 'Evidence / Clause', color: '#22d3ee', radius: 7 },
  risk: { label: 'Gap / Risk', color: '#fb7185', radius: 10 },
  recommendation: { label: 'Recommendation', color: '#fb923c', radius: 9 },
}

function isHashLike(value: string): boolean {
  return /^[a-f0-9]{12,}$/i.test(value.replace(/[^a-f0-9]/gi, ''))
}

function cleanGraphLabel(raw: string): string {
  const label = (raw || '').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim()
  if (!label) return 'Compliance Evidence'
  if (isHashLike(label)) return 'Policy Evidence'

  const lower = label.toLowerCase()
  if (lower.includes('gdpr')) return 'GDPR Requirement'
  if (lower.includes('hipaa')) return 'HIPAA Requirement'
  if (lower.includes('ccpa')) return 'CCPA Requirement'
  if (lower.includes('nist')) return 'NIST Control'
  if (lower.includes('gap') || lower.includes('missing')) return 'Compliance Gap'
  if (lower.includes('recommend')) return 'Recommended Fix'

  return label.length > 46 ? `${label.slice(0, 43)}...` : label
}

function categorizeNode(label: string): NodeCategory {
  const lower = label.toLowerCase()
  if (lower.includes('gap') || lower.includes('risk') || lower.includes('missing') || lower.includes('conflict')) return 'risk'
  if (lower.includes('recommend') || lower.includes('remediation')) return 'recommendation'
  if (lower.includes('policy') || lower.endsWith('.pdf')) return 'policy'
  if (lower.includes('gdpr') || lower.includes('hipaa') || lower.includes('ccpa') || lower.includes('nist') || lower.includes('regulation')) return 'regulation'
  if (lower.includes('requirement') || lower.includes('obligation') || lower.includes('control') || lower.includes('article')) return 'requirement'
  return 'evidence'
}

export default function GraphExplorer() {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [data, setData]       = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)

  const load = () => {
    setLoading(true); setError(null)
    graphAPI(80)
      .then(d => {
        if (!d || !Array.isArray(d.nodes) || !Array.isArray(d.links)) {
          console.error('[GraphExplorer] unexpected /graph response:', d)
          throw new Error(
            `Unexpected response shape from /graph (got ${typeof d}` +
            (d && typeof d === 'object' ? `, keys: ${Object.keys(d).join(', ')}` : '') +
            ') — see browser console for the raw response.'
          )
        }
        setData(d)
      })
      .catch((e) => setError(e?.message ? `Could not load the knowledge graph: ${e.message}` : 'Could not load the knowledge graph from Neo4j.'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  useEffect(() => {
    if (!data || !svgRef.current) return

    try {
      const width = svgRef.current.clientWidth || 900
      const height = 620

      const connectedIds = new Set<string>()
      data.links.forEach(l => {
        connectedIds.add(String(l.source))
        connectedIds.add(String(l.target))
      })

      const nodes: D3Node[] = data.nodes
        .filter(n => connectedIds.has(n.id))
        .slice(0, 45)
        .map(n => {
          const displayLabel = cleanGraphLabel(n.label || n.id)
          return {
            ...n,
            displayLabel,
            category: categorizeNode(`${displayLabel} ${n.label || ''} ${n.id || ''}`),
          }
        })
      const nodeIds = new Set(nodes.map(n => n.id))
      // Drop edges that reference a node not present in `nodes` — d3.forceLink
      // throws synchronously if source/target can't be resolved.
      const links: D3Link[] = data.links
        .filter(l => nodeIds.has(l.source as unknown as string) && nodeIds.has(l.target as unknown as string))
        .map(l => ({ ...l })) as D3Link[]

      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()
      svg.attr('viewBox', `0 0 ${width} ${height}`)

      const container = svg.append('g')
      svg.call(
        d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.2, 4])
          .on('zoom', (e) => container.attr('transform', e.transform)) as any
      )

      const sim = d3.forceSimulation<D3Node>(nodes)
        .force('link', d3.forceLink<D3Node, D3Link>(links).id(d => d.id).distance(130))
        .force('charge', d3.forceManyBody().strength(-420))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide(42))

      const link = container.append('g').attr('stroke', '#666').attr('stroke-opacity', 0.5)
        .selectAll('line').data(links).join('line').attr('stroke-width', 1.6)

      link.append('title').text(d => d.type + (d.description ? ` — ${d.description}` : ''))

      const color = (t: string) =>
        t === 'STRICTER_THAN' ? '#f58231' : t === 'CONFLICTS_WITH' ? '#e6194B' : '#4363d8'
      link.attr('stroke', d => color(d.type))

      const node = container.append('g').attr('stroke', '#1a1a2e').attr('stroke-width', 1.5)
        .selectAll('circle').data(nodes).join('circle')
        .attr('r', d => CATEGORY_META[d.category].radius)
        .attr('fill', d => CATEGORY_META[d.category].color)
        .call(
          d3.drag<SVGCircleElement, D3Node>()
            .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
            .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
            .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null }) as any
        )
      node.append('title').text(d => `${CATEGORY_META[d.category].label}: ${d.label || d.id}`)

      const label = container.append('g')
        .selectAll('text').data(nodes).join('text')
        .text(d => (d.label || d.id || '').length > 22 ? (d.label || d.id).slice(0, 22) + '…' : (d.label || d.id))
        .attr('font-size', 9).attr('fill', '#cbd5e1').attr('dx', 10).attr('dy', 3)

      sim.on('tick', () => {
        link
          .attr('x1', d => (d.source as D3Node).x!).attr('y1', d => (d.source as D3Node).y!)
          .attr('x2', d => (d.target as D3Node).x!).attr('y2', d => (d.target as D3Node).y!)
        node.attr('cx', d => d.x!).attr('cy', d => d.y!)
        label.attr('x', d => d.x!).attr('y', d => d.y!)
      })

      return () => { sim.stop() }
    } catch (e: any) {
      setError(`Could not render the graph: ${e?.message || e}`)
      return undefined
    }
  }, [data])

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="page-title">Graph Explorer</div>
          <div className="page-sub">Demo-friendly view of the compliance knowledge graph: frameworks, requirements, evidence, and risks</div>
        </div>
        <button className="btn btn-ghost" onClick={load} style={{ fontSize: 12 }}><RefreshCw size={12} /> Reload</button>
      </div>

      {error && (
        <div style={{ background: '#3b0e0e', border: '1px solid var(--critical)', borderRadius: 8, padding: 14, marginBottom: 16, color: 'var(--critical)', fontSize: 13 }}>
          {error}
        </div>
      )}

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        {loading ? (
          <div className="empty-state"><span className="spinner" /> Loading graph…</div>
        ) : !data || data.nodes.length === 0 ? (
          <div className="empty-state"><GitBranch size={32} /> Graph is empty — build the KG first.</div>
        ) : (
          <>
            <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)', fontSize: 12, color: 'var(--text-dim)' }}>
              Showing a simplified graph for presentation: long chunk IDs are renamed as <strong style={{ color: 'var(--text)' }}>Policy Evidence</strong>,
              and the graph is limited to the most connected nodes so the compliance story is easier to follow.
            </div>
            <svg ref={svgRef} style={{ width: '100%', height: 620, background: '#1a1a2e', display: 'block' }} />
          </>
        )}
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 10, fontSize: 11, color: 'var(--text-muted)' }}>
        <span><span style={{ color: '#e6194B' }}>■</span> CONFLICTS_WITH</span>
        <span><span style={{ color: '#f58231' }}>■</span> STRICTER_THAN</span>
        <span><span style={{ color: '#4363d8' }}>■</span> other</span>
      </div>
    </div>
  )
}
