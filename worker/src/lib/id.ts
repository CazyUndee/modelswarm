/**
 * ID generation utilities
 */

export function generateAgentId(): string {
  const suffix = Array.from(crypto.getRandomValues(new Uint8Array(2)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return `agent-${suffix}`;
}

export function generateApiKey(): string {
  const bytes = Array.from(crypto.getRandomValues(new Uint8Array(32)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return `ms_${bytes}`;
}

export function generateExperimentId(counter: number): string {
  return `EXP-${counter.toString().padStart(3, '0')}`;
}

export function generatePostId(): string {
  const suffix = Array.from(crypto.getRandomValues(new Uint8Array(4)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return `POST-${suffix}`;
}

export function generateScriptId(): string {
  const suffix = Array.from(crypto.getRandomValues(new Uint8Array(4)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return `SCRIPT-${suffix}`;
}

export function generateNotificationId(): string {
  const suffix = Array.from(crypto.getRandomValues(new Uint8Array(4)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return `NOTIF-${suffix}`;
}

export function generateCommentId(): string {
  const suffix = Array.from(crypto.getRandomValues(new Uint8Array(4)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return `COMMENT-${suffix}`;
}

export function generateRunnerId(): string {
  const suffix = Array.from(crypto.getRandomValues(new Uint8Array(4)))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
  return `RUNNER-${suffix}`;
}

export function now(): string {
  return new Date().toISOString();
}
