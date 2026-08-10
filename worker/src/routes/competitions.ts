/**
 * Competitions API routes
 * GET  /api/competitions
 * GET  /api/competitions/:id
 * GET  /api/competitions/:id/state
 * POST /api/competitions/:id/join
 * GET  /api/competitions/:id/agents
 */

import { Env } from '../types';
import { ok, notFound } from '../lib/json';
import { now } from '../lib/id';

export async function handleListCompetitions(env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT * FROM competitions ORDER BY created_at DESC')
    .all();
  return ok({ competitions: result.results });
}

export async function handleGetCompetition(competitionId: string, env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT * FROM competitions WHERE competition_id = ?')
    .bind(competitionId)
    .first();

  if (!result) return notFound('Competition');
  return ok(result);
}

export async function handleGetCompetitionState(competitionId: string, env: Env): Promise<Response> {
  // Get competition info
  const comp = await env.DB
    .prepare('SELECT * FROM competitions WHERE competition_id = ?')
    .bind(competitionId)
    .first();

  if (!comp) return notFound('Competition');

  // Get current champion (best OOF score)
  const champion = await env.DB
    .prepare('SELECT experiment_id, oof_metric FROM experiments WHERE competition_id = ? AND oof_metric IS NOT NULL ORDER BY oof_metric DESC LIMIT 1')
    .bind(competitionId)
    .first<{ experiment_id: string; oof_metric: number }>();

  // Count experiments by status
  const statusCounts = await env.DB
    .prepare('SELECT status, COUNT(*) as count FROM experiments WHERE competition_id = ? GROUP BY status')
    .bind(competitionId)
    .all();

  return ok({
    competition: comp,
    champion_experiment_id: champion?.experiment_id || null,
    best_score: champion?.oof_metric || null,
    experiment_counts: statusCounts.results,
    current_phase: 4, // Default; would be derived from actual state
  });
}

export async function handleJoinCompetition(request: Request, env: Env, competitionId: string): Promise<Response> {
  const body = await request.json<{ agent_id: string }>();
  const timestamp = now();

  await env.DB
    .prepare('INSERT OR IGNORE INTO agent_competitions (agent_id, competition_id, joined_at) VALUES (?, ?, ?)')
    .bind(body.agent_id, competitionId, timestamp)
    .run();

  return ok({ agent_id: body.agent_id, competition_id: competitionId, status: 'joined', joined_at: timestamp });
}

export async function handleGetCompetitionAgents(competitionId: string, env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT a.agent_id, a.name, a.model, a.role, a.status FROM agents a JOIN agent_competitions ac ON a.agent_id = ac.agent_id WHERE ac.competition_id = ?')
    .bind(competitionId)
    .all();

  return ok({ agents: result.results });
}
