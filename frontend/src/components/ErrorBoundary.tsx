// src/components/ErrorBoundary.tsx
import { Component, ReactNode } from 'react'
import { AlertTriangle } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

// Catches render-time crashes in a page (e.g. malformed API data) so a single
// page failure shows an error message instead of taking down the whole app
// with a blank screen.
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: { componentStack?: string }) {
    console.error('Page crashed:', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="card" style={{ borderColor: 'var(--critical)' }}>
          <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--critical)' }}>
            <AlertTriangle size={14} /> Something went wrong rendering this page
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', whiteSpace: 'pre-wrap' }}>
            {this.state.error.message}
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
