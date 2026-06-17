// src/pages/GraphExplorer.tsx
import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { graphAPI, GraphData } from '../lib/api'
import { GitBranch, RefreshCw } from 'lucide-react'

interface D3Node extends d3.SimulationNodeDatum { id: string; label: string }
interface D3Link extends d3.SimulationLinkDatum<D3Node> { type: string; description?: string }

export default function GraphExplorer() {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [data, setData]       = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)

  const load = () => {
    setLoading(true); setError(null)
    graphAPI(200)
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

      const nodes: D3Node[] = data.nodes.map(n => ({ ...n }))
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
        .force('link', d3.forceLink<D3Node, D3Link>(links).id(d => d.id).distance(90))
        .force('charge', d3.forceManyBody().strength(-220))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide(22))

      const link = container.append('g').attr('stroke', '#666').attr('stroke-opacity', 0.5)
        .selectAll('line').data(links).join('line').attr('stroke-width', 1.2)

      link.append('title').text(d => d.type + (d.description ? ` — ${d.description}` : ''))

      const color = (t: string) =>
        t === 'STRICTER_THAN' ? '#f58231' : t === 'CONFLICTS_WITH' ? '#e6194B' : '#4363d8'
      link.attr('stroke', d => color(d.type))

      const node = container.append('g').attr('stroke', '#1a1a2e').attr('stroke-width', 1.5)
        .selectAll('circle').data(nodes).join('circle')
        .attr('r', 7).attr('fill', '#42d4f4')
        .call(
          d3.drag<SVGCircleElement, D3Node>()
            .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
            .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
            .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null }) as any
        )
      node.append('title').text(d => d.label)

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
          <div className="page-sub">Interactive view of the compliance knowledge graph (Neo4j)</div>
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
          <svg ref={svgRef} style={{ width: '100%', height: 620, background: '#1a1a2e', display: 'block' }} />
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
