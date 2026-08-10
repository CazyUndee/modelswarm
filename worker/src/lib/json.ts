/**
 * JSON response helpers
 */

export function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

export function error(message: string, status = 400): Response {
  return json({ error: message }, status);
}

export function notFound(resource = 'Resource'): Response {
  return error(`${resource} not found`, 404);
}

export function unauthorized(message = 'Unauthorized'): Response {
  return error(message, 401);
}

export function conflict(message = 'Conflict'): Response {
  return error(message, 409);
}

export function created(data: unknown): Response {
  return json(data, 201);
}

export function ok(data: unknown = { success: true }): Response {
  return json(data, 200);
}
