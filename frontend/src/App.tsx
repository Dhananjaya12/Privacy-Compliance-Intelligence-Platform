// src/App.tsx
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Shield, Search, BarChart2, GitBranch, History as HistoryIcon } from 'lucide-react'
import Dashboard     from './pages/Dashboard'
import Audit         from './pages/Audit'
import History       from './pages/History'
import GraphExplorer from './pages/GraphExplorer'
import ErrorBoundary from './components/ErrorBoundary'
import './index.css'

const NAV = [
  { to: '/',            icon: BarChart2,   label: 'Dashboard'   },
  { to: '/audit',       icon: Search,      label: 'Audit'       },
  { to: '/graph',       icon: GitBranch,   label: 'Graph'       },
  { to: '/history',     icon: HistoryIcon, label: 'History'     },
]

function MainContent() {
  // Re-key the error boundary per route so navigating away from a crashed
  // page recovers instead of staying blank.
  const location = useLocation()
  return (
    <ErrorBoundary key={location.pathname}>
      <Routes>
        <Route path="/"            element={<Dashboard />} />
        <Route path="/audit"       element={<Audit />} />
        <Route path="/graph"       element={<GraphExplorer />} />
        <Route path="/history"     element={<History />} />
      </Routes>
    </ErrorBoundary>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">

        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <div className="sidebar-brand">
            <Shield size={22} strokeWidth={1.5} />
            <span>PrivacyGuard</span>
          </div>

          <nav className="sidebar-nav">
            {NAV.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              >
                <Icon size={16} strokeWidth={1.5} />
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="sidebar-footer">
            <span className="version-tag">Compliance · v1.0</span>
          </div>
        </aside>

        {/* ── Main content ── */}
        <main className="main-content">
          <MainContent />
        </main>

      </div>
    </BrowserRouter>
  )
}
