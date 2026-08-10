/**
 * Shared Scripts API routes
 * GET  /api/scripts
 * POST /api/scripts
 * GET  /api/scripts/:id
 */

import { Env } from '../types';
import { ok, error, notFound, created } from '../lib/json';
import { generateScriptId, now } from '../lib/id';

export async function handleListScripts(env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT * FROM shared_scripts ORDER BY created_at DESC')
    .all();

  return ok({ scripts: result.results });
}

export async function handleGetScript(scriptId: string, env: Env): Promise<Response> {
  const result = await env.DB
    .prepare('SELECT * FROM shared_scripts WHERE script_id = ?')
    .bind(scriptId)
    .first();

  if (!result) return notFound('Script');
  return ok(result);
}

export async function handlePublishScript(request: Request, env: Env): Promise<Response> {
  const body = await request.json<{
    name: string; source_path: string; description?: string;
    author_id?: string; version?: string; dependencies?: string[]; usage?: string;
  }>();

  if (!body.name || !body.source_path) {
    return error('name and source_path are required');
  }

  const scriptId = generateScriptId();
  const timestamp = now();
  const dependencies = JSON.stringify(body.dependencies || []);

  await env.DB
    .prepare(`INSERT INTO shared_scripts (script_id, name, author_id, description, version, dependencies, usage, source_path, created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(scriptId, body.name, body.author_id || 'system', body.description || '', body.version || '1.0.0', dependencies, body.usage || null, body.source_path, timestamp, timestamp)
    .run();

  return created({ script_id: scriptId, name: body.name, created_at: timestamp });
}
