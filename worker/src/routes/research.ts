/**
 * Research API routes
 * GET  /api/research/state
 * GET  /api/research/findings
 * POST /api/research/findings
 */

import { Env } from '../types';
import { ok, error, created } from '../lib/json';
import { generatePostId, now } from '../lib/id';

export async function handleGetResearchState(env: Env): Promise<Response> {
  // Aggregate research state from database
  const champion = await env.DB
    .prepare('SELECT experiment_id, oof_metric FROM experiments WHERE oof_metric IS NOT NULL ORDER BY oof_metric DESC LIMIT 1')
    .first<{ experiment_id: string; oof_metric: number }>();

  const activeExps = await env.DB
    .prepare("SELECT COUNT(*) as count FROM experiments WHERE status IN ('claimed', 'active')")
    .first<{ count: number }>();

  const queuedExps = await env.DB
    .prepare("SELECT COUNT(*) as count FROM experiments WHERE status = 'queued'")
    .first<{ count: number }>();

  return ok({
    current_phase: 4,
    champion: champion?.experiment_id || null,
    best_score: champion?.oof_metric || null,
    active_experiments: activeExps?.count || 0,
    queued_experiments: queuedExps?.count || 0,
  });
}

export async function handleListFindings(env: Env): Promise<Response> {
  const result = await env.DB
    .prepare("SELECT * FROM forum_posts WHERE category = 'discovery' ORDER BY created_at DESC LIMIT 50")
    .all();

  return ok({ findings: result.results });
}

export async function handlePublishFinding(request: Request, env: Env): Promise<Response> {
  const body = await request.json<{ title: string; content: string; author_id?: string; experiment_id?: string }>();

  if (!body.title || !body.content) {
    return error('title and content are required');
  }

  const postId = generatePostId();
  const timestamp = now();

  await env.DB
    .prepare(`INSERT INTO forum_posts (post_id, author_id, category, title, content, experiment_id, created_at)
              VALUES (?, ?, 'discovery', ?, ?, ?, ?)`)
    .bind(postId, body.author_id || 'system', body.title, body.content, body.experiment_id || null, timestamp)
    .run();

  return created({ post_id: postId, category: 'discovery', created_at: timestamp });
}
