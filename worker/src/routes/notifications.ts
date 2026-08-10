/**
 * Notifications API routes
 * GET  /api/notifications
 * POST /api/notifications/mark-read
 */

import { Env } from '../types';
import { ok, error } from '../lib/json';
import { generateNotificationId, now } from '../lib/id';

export async function handleGetNotifications(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const agentId = url.searchParams.get('agent_id') || 'system';
  const unreadOnly = url.searchParams.get('unread') === 'true';

  let query = 'SELECT * FROM notifications WHERE recipient_id = ?';
  if (unreadOnly) query += ' AND read = 0';
  query += ' ORDER BY created_at DESC LIMIT 50';

  const result = await env.DB.prepare(query).bind(agentId).all();
  return ok({ notifications: result.results });
}

export async function handleMarkRead(request: Request, env: Env): Promise<Response> {
  const body = await request.json<{ notification_id: string }>();

  if (!body.notification_id) {
    return error('notification_id is required');
  }

  await env.DB
    .prepare('UPDATE notifications SET read = 1 WHERE notification_id = ?')
    .bind(body.notification_id)
    .run();

  return ok({ notification_id: body.notification_id, read: true });
}
