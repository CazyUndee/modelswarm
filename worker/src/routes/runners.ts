/**
 * Runners (compute) API routes
 * GET  /api/runners
 * POST /api/runners
 * POST /api/runners/:id/heartbeat
 */

import { Env } from '../types';
import { ok, error, created } from '../lib/json';
import { generateRunnerId, now } from '../lib/id';

export async function handleListRunners(env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT * FROM runners ORDER BY registered_at DESC')
    .all();

  return ok({ runners: result.results });
}

export async function handleRegisterRunner(request: Request, env: Env): Promise<Response> {
  const body = await request.json<{ name: string; provider: string; capabilities?: Record<string, unknown> }>();

  if (!body.name || !body.provider) {
    return error('name and provider are required');
  }

  const runnerId = generateRunnerId();
  const timestamp = now();
  const capabilities = JSON.stringify(body.capabilities || {});

  await env.DB
    .prepare('INSERT INTO runners (runner_id, name, provider, capabilities, status, registered_at) VALUES (?, ?, ?, ?, ?, ?)')
    .bind(runnerId, body.name, body.provider, capabilities, 'available', timestamp)
    .run();

  return created({ runner_id: runnerId, status: 'available', registered_at: timestamp });
}

export async function handleRunnerHeartbeat(runnerId: string, env: Env): Promise<Response> {
  const timestamp = now();

  await env.DB
    .prepare('UPDATE runners SET last_heartbeat = ? WHERE runner_id = ?')
    .bind(timestamp, runnerId)
    .run();

  return ok({ runner_id: runnerId, last_heartbeat: timestamp });
}
