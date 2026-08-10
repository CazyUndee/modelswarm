/**
 * Agents API routes
 * POST /api/agents/register
 * GET  /api/agents
 * GET  /api/agents/:id
 * POST /api/agents/:id/heartbeat
 * POST /api/agents/:id/status
 * GET  /api/agents/:id/subagents
 * POST /api/agents/:id/subagents
 */

import { Env } from '../types';
import { ok, error, notFound, created } from '../lib/json';
import { generateAgentId, generateApiKey, now } from '../lib/id';

export async function handleRegisterAgent(request: Request, env: Env): Promise<Response> {
  const body = await request.json<{ name: string; model: string; role?: string; parent_agent_id?: string }>();

  if (!body.name || !body.model) {
    return error('name and model are required');
  }

  const agentId = generateAgentId();
  const apiKey = generateApiKey();
  const role = body.role || 'research';
  const timestamp = now();

  await env.DB
    .prepare(`INSERT INTO agents (agent_id, name, model, role, parent_agent_id, api_key, registered_at, status)
              VALUES (?, ?, ?, ?, ?, ?, ?, 'active')`)
    .bind(agentId, body.name, body.model, role, body.parent_agent_id || null, apiKey, timestamp)
    .run();

  return created({ agent_id: agentId, api_key: apiKey, registered_at: timestamp });
}

export async function handleListAgents(env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT agent_id, name, model, role, parent_agent_id, registered_at, last_heartbeat, status FROM agents ORDER BY registered_at DESC')
    .all();

  return ok({ agents: result.results });
}

export async function handleGetAgent(agentId: string, env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT agent_id, name, model, role, parent_agent_id, registered_at, last_heartbeat, status, capabilities FROM agents WHERE agent_id = ?')
    .bind(agentId)
    .first();

  if (!result) return notFound('Agent');
  return ok(result);
}

export async function handleHeartbeat(agentId: string, env: Env): Promise<Response> {
  const timestamp = now();
  const result = await env.DB
    .prepare('UPDATE agents SET last_heartbeat = ? WHERE agent_id = ?')
    .bind(timestamp, agentId)
    .run();

  if (!result.success) return notFound('Agent');
  return ok({ agent_id: agentId, last_heartbeat: timestamp });
}

export async function handleUpdateAgentStatus(request: Request, agentId: string, env: Env): Promise<Response> {
  const body = await request.json<{ status: string }>();
  const validStatuses = ['active', 'idle', 'stalled', 'terminated'];

  if (!validStatuses.includes(body.status)) {
    return error(`Invalid status. Must be one of: ${validStatuses.join(', ')}`);
  }

  const result = await env.DB
    .prepare('UPDATE agents SET status = ? WHERE agent_id = ?')
    .bind(body.status, agentId)
    .run();

  if (!result.success) return notFound('Agent');
  return ok({ agent_id: agentId, status: body.status });
}

export async function handleListSubagents(agentId: string, env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT agent_id, name, model, role, registered_at, status FROM agents WHERE parent_agent_id = ?')
    .bind(agentId)
    .all();

  return ok({ subagents: result.results });
}

export async function handleCreateSubagent(request: Request, agentId: string, env: Env): Promise<Response> {
  const body = await request.json<{ name: string; model: string; role?: string }>();

  if (!body.name || !body.model) {
    return error('name and model are required');
  }

  const parent = await env.DB
    .prepare('SELECT agent_id FROM agents WHERE agent_id = ?')
    .bind(agentId)
    .first();

  if (!parent) return notFound('Parent agent');

  const existing = await env.DB
    .prepare('SELECT COUNT(*) as count FROM agents WHERE parent_agent_id = ?')
    .bind(agentId)
    .first<{ count: number }>();

  const subagentNumber = (existing?.count || 0) + 1;
  const subagentId = `${agentId}-${subagentNumber.toString().padStart(2, '0')}`;
  const apiKey = generateApiKey();
  const role = body.role || 'research';
  const timestamp = now();

  await env.DB
    .prepare(`INSERT INTO agents (agent_id, name, model, role, parent_agent_id, api_key, registered_at, status)
              VALUES (?, ?, ?, ?, ?, ?, ?, 'active')`)
    .bind(subagentId, body.name, body.model, role, agentId, apiKey, timestamp)
    .run();

  return created({ agent_id: subagentId, api_key: apiKey, parent_agent_id: agentId, registered_at: timestamp });
}
