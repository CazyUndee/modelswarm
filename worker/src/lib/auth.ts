/**
 * Authentication utilities
 */

import { Env } from '../types';
import { error, unauthorized } from './json';

/**
 * Extract and verify API key from request headers.
 * Returns the agent_id if valid, null otherwise.
 */
export async function authenticate(request: Request, env: Env): Promise<string | null> {
  const authHeader = request.headers.get('Authorization');
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return null;
  }

  const apiKey = authHeader.slice(7);

  // Check against admin key for system operations
  if (env.MODELSWARM_ADMIN_KEY && apiKey === env.MODELSWARM_ADMIN_KEY) {
    return 'admin';
  }

  // Look up agent by API key
  const result = await env.DB
    .prepare('SELECT agent_id FROM agents WHERE api_key = ?')
    .bind(apiKey)
    .first<{ agent_id: string }>();

  return result?.agent_id ?? null;
}

/**
 * Require authentication. Returns agent_id or throws a response.
 */
export async function requireAuth(request: Request, env: Env): Promise<string | Response> {
  const agentId = await authenticate(request, env);
  if (!agentId) {
    return unauthorized('Invalid or missing API key');
  }
  return agentId;
}
