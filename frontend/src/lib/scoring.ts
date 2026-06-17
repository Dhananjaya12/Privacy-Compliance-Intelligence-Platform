// src/lib/scoring.ts
//
// Client-side port of the compliance scoring formula in
// agent/compliance_nodes.py (risk_scorer_node / risk_to_compliance).
// Used to project a "what if these findings were resolved" score as the
// user ticks off remediation checklist items. This is a projection/estimate
// only — not an official re-audit.

import { GapGroup } from './api'

export const SEVERITY_SCORES: Record<string, number> = {
  critical: 10.0,
  high: 7.5,
  medium: 5.0,
  low: 2.5,
  info: 1.0,
}

export const REGULATION_WEIGHTS: Record<string, number> = {
  GDPR: 0.35,
  HIPAA: 0.30,
  CCPA: 0.20,
  NIST: 0.15,
}

const DEFAULT_REGULATION_WEIGHT = 0.1

// Convert internal risk (0-10, higher=worse) to compliance (0-100, higher=better).
export function riskToCompliance(risk0to10: number): number {
  return Math.round(Math.max(0, Math.min(100, 100 - risk0to10 * 10)) * 10) / 10
}

export interface ProjectedScore {
  overall: number | null
  perReg: Record<string, number>
}

/**
 * Recompute compliance scores as if the gap groups whose theme is in
 * `resolvedThemes` had been fully addressed (their gaps no longer count
 * against the score).
 */
export function projectCompliance(
  gapGroups: GapGroup[],
  obligationCounts: Record<string, number>,
  resolvedThemes: Set<string>,
): ProjectedScore {
  const regGapWeight: Record<string, number> = {}
  for (const group of gapGroups) {
    if (resolvedThemes.has(group.theme)) continue
    for (const gap of group.gaps) {
      const reg = gap.regulation
      regGapWeight[reg] = (regGapWeight[reg] ?? 0) + (SEVERITY_SCORES[gap.severity] ?? 5.0)
    }
  }

  const riskScores: Record<string, number> = {}
  for (const [reg, obCount] of Object.entries(obligationCounts)) {
    if (!obCount) continue
    const raw = regGapWeight[reg] ?? 0
    riskScores[reg] = Math.min((raw / (obCount * 10.0)) * 10.0, 10.0)
  }

  const scoredRegs = Object.keys(riskScores)
  const totalWeight = scoredRegs.reduce(
    (sum, reg) => sum + (REGULATION_WEIGHTS[reg] ?? DEFAULT_REGULATION_WEIGHT), 0,
  )

  let overallRisk = 0
  if (totalWeight > 0 && scoredRegs.length > 0) {
    for (const reg of scoredRegs) {
      const w = (REGULATION_WEIGHTS[reg] ?? DEFAULT_REGULATION_WEIGHT) / totalWeight
      overallRisk += w * riskScores[reg]
    }
  }

  const perReg: Record<string, number> = {}
  for (const reg of scoredRegs) perReg[reg] = riskToCompliance(riskScores[reg])

  return {
    overall: scoredRegs.length > 0 ? riskToCompliance(overallRisk) : null,
    perReg,
  }
}
