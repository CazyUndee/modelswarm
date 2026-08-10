/**
 * Experiments API routes
 * GET  /api/experiments
 * POST /api/experiments
 * GET  /api/experiments/:id
 * POST /api/experiments/:id/claim
 * POST /api/experiments/:id/complete
 * POST /api/experiments/:id/fail
 */

import { Env } from '../types';
import { ok, error, notFound, created, conflict } from '../lib/json';
import { now } from '../lib/id';

export async function handleListExperiments(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const status = url.searchParams.get('status');
  const agentId = url.searchParams.get('agent_id');
  const competitionId = url.searchParams.get('competition_id');
  const phase = url.searchParams.get('phase');

  let query = 'SELECT * FROM experiments WHERE 1=1';
  const params: (string | number)[] = [];

  if (status) { query += ' AND status = ?'; params.push(status); }
  if (agentId) { query += ' AND agent_id = ?'; params.push(agentId); }
  if (competitionId) { query += ' AND competition_id = ?'; params.push(competitionId); }
  if (phase) { query += ' AND phase = ?'; params.push(parseInt(phase)); }

  query += ' ORDER BY created_at DESC';

  const result = await env.DB.prepare(query).bind(...params).all();
  return ok({ experiments: result.results });
}

export async function handleCreateExperiment(request: Request, env: Env): Promise<Response> {
  const body = await request.json<Record<string, unknown>>();

  if (!body.hypothesis) {
    return error('hypothesis is required');
  }

  const countResult = await env.DB
    .prepare('SELECT COUNT(*) as count FROM experiments')
    .first<{ count: number }>();
  const nextId = `EXP-${((countResult?.count || 0)).toString().padStart(3, '0')}`;

  const timestamp = now();
  const features = JSON.stringify(body.features || []);
  const configuration = JSON.stringify(body.configuration || {});
  const foldMetrics = JSON.stringify(body.fold_metrics || []);
  const artifacts = JSON.stringify(body.artifacts || []);
  const computeInfo = JSON.stringify(body.compute_info || {});

  await env.DB
    .prepare(`INSERT INTO experiments
      (experiment_id, hypothesis, agent_id, executing_agent_id, parent_agent_id, parent_experiment_id,
       competition_id, phase, configuration, dataset, features, model, validation_protocol,
       fold_metrics, compute_info, artifacts, status, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)`)
    .bind(
      nextId, body.hypothesis, body.agent_id || null, body.executing_agent_id || null,
      body.parent_agent_id || null, body.parent_experiment_id || null,
      body.competition_id || null, body.phase || null, configuration,
      body.dataset || null, features, body.model || null, body.validation_protocol || null,
      foldMetrics, computeInfo, artifacts, timestamp
    )
    .run();

  return created({ experiment_id: nextId, status: 'queued', created_at: timestamp });
}

export async function handleGetExperiment(experimentId: string, env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT * FROM experiments WHERE experiment_id = ?')
    .bind(experimentId)
    .first();

  if (!result) return notFound('Experiment');
  return ok(result);
}

export async function handleClaimExperiment(request: Request, env: Env, experimentId: string): Promise<Response> {
  const body = await request.json<{ agent_id: string; release?: boolean }>();
  const claimExpiryMinutes = parseInt(env.CLAIM_EXPIRY_MINUTES || '30');
  const expiryTime = new Date(Date.now() - claimExpiryMinutes * 60000).toISOString();

  if (body.release) {
    await env.DB
      .prepare("UPDATE experiments SET status = 'queued', claimed_by = NULL, claimed_at = NULL WHERE experiment_id = ?")
      .bind(experimentId)
      .run();
    return ok({ experiment_id: experimentId, status: 'queued', released: true });
  }

  // Exclusive lock: only claim if queued or expired claim
  const timestamp = now();
  const result = await env.DB
    .prepare(`UPDATE experiments SET status = 'claimed', claimed_by = ?, claimed_at = ?
              WHERE experiment_id = ? AND (status = 'queued' OR (status = 'claimed' AND claimed_at < ?))`)
    .bind(body.agent_id, timestamp, experimentId, expiryTime)
    .run();

  if (!result.meta.changes) {
    return conflict('Experiment already claimed or not available');
  }

  return ok({ experiment_id: experimentId, status: 'claimed', claimed_by: body.agent_id, claimed_at: timestamp });
}

export async function handleCompleteExperiment(request: Request, env: Env, experimentId: string): Promise<Response> {
  const body = await request.json<Record<string, unknown>>();
  const timestamp = now();

  const foldMetrics = JSON.stringify(body.fold_metrics || []);
  const artifacts = JSON.stringify(body.artifacts || []);

  const result = await env.DB
    .prepare(`UPDATE experiments SET status = 'completed', oof_metric = ?, fold_metrics = ?,
              public_score = ?, runtime_seconds = ?, decision = ?, reasoning = ?,
              artifacts = ?, completed_at = ?, executing_agent_id = ?
              WHERE experiment_id = ?`)
    .bind(
      body.oof_metric ?? null, foldMetrics, body.public_score ?? null,
      body.runtime_seconds ?? null, body.decision ?? null, body.reasoning ?? null,
      artifacts, timestamp, body.executing_agent_id ?? null, experimentId
    )
    .run();

  if (!result.success) return notFound('Experiment');
  return ok({ experiment_id: experimentId, status: 'completed', completed_at: timestamp });
}

export async function handleFailExperiment(request: Request, env: Env, experimentId: string): Promise<Response> {
  const body = await request.json<{ reason: string }>();
  const timestamp = now();

  const result = await env.DB
    .prepare(`UPDATE experiments SET status = 'failed', reasoning = ?, completed_at = ? WHERE experiment_id = ?`)
    .bind(body.reasoning || body.reason || 'Unknown failure', timestamp, experimentId)
    .run();

  if (!result.success) return notFound('Experiment');
  return ok({ experiment_id: experimentId, status: 'failed', completed_at: timestamp });
}
