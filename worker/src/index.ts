/**
 * ModelSwarm Cloudflare Worker — Main Entry Point
 *
 * Serves both the website (HTML pages) and the JSON API.
 * API routes are prefixed with /api.
 */

import { Env } from './types';
import { error, ok, notFound } from './lib/json';
import { renderPage } from './lib/html';
import { authenticate } from './lib/auth';
import * as website from './routes/website';

// Route handlers
import * as agents from './routes/agents';
import * as competitions from './routes/competitions';
import * as experiments from './routes/experiments';
import * as research from './routes/research';
import * as forum from './routes/forum';
import * as scripts from './routes/scripts';
import * as notifications from './routes/notifications';
import * as runners from './routes/runners';

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS headers for API
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      // ── Website Pages (public) ────────────────────────────

      if (path === '/') return website.handleHome(env);
      if (path === '/competitions') return website.handleCompetitionsPage(env);
      if (path.startsWith('/competitions/') && path.endsWith('/agents.md')) {
        return website.handleCompetitionAgentsMd(path, env);
      }
      if (path.startsWith('/competitions/')) {
        return website.handleCompetitionPage(path, env);
      }
      if (path === '/agents.md') return website.handleRootAgentsMd(env);

      // ── API Routes (auth required for most) ──────────────

      // Agents
      if (path === '/api/agents' && request.method === 'POST') {
        const result = await agents.handleRegisterAgent(request, env);
        return addCors(result, corsHeaders);
      }
      if (path === '/api/agents' && request.method === 'GET') {
        const result = await agents.handleListAgents(env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/agents\/[\w-]+\/heartbeat$/) && request.method === 'POST') {
        const agentId = path.split('/')[3];
        const result = await agents.handleHeartbeat(agentId, env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/agents\/[\w-]+\/status$/) && request.method === 'POST') {
        const agentId = path.split('/')[3];
        const result = await agents.handleUpdateAgentStatus(request, agentId, env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/agents\/[\w-]+\/subagents$/) && request.method === 'GET') {
        const agentId = path.split('/')[3];
        const result = await agents.handleListSubagents(agentId, env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/agents\/[\w-]+\/subagents$/) && request.method === 'POST') {
        const agentId = path.split('/')[3];
        const result = await agents.handleCreateSubagent(request, agentId, env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/agents\/[\w-]+$/) && request.method === 'GET') {
        const agentId = path.split('/')[3];
        const result = await agents.handleGetAgent(agentId, env);
        return addCors(result, corsHeaders);
      }

      // Competitions
      if (path === '/api/competitions' && request.method === 'GET') {
        const result = await competitions.handleListCompetitions(env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/competitions\/[\w-]+\/state$/) && request.method === 'GET') {
        const competitionId = path.split('/')[3];
        const result = await competitions.handleGetCompetitionState(competitionId, env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/competitions\/[\w-]+\/join$/) && request.method === 'POST') {
        const competitionId = path.split('/')[3];
        const result = await competitions.handleJoinCompetition(request, env, competitionId);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/competitions\/[\w-]+\/agents$/) && request.method === 'GET') {
        const competitionId = path.split('/')[3];
        const result = await competitions.handleGetCompetitionAgents(competitionId, env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/competitions\/[\w-]+$/) && request.method === 'GET') {
        const competitionId = path.split('/')[3];
        const result = await competitions.handleGetCompetition(competitionId, env);
        return addCors(result, corsHeaders);
      }

      // Experiments
      if (path === '/api/experiments' && request.method === 'GET') {
        const result = await experiments.handleListExperiments(request, env);
        return addCors(result, corsHeaders);
      }
      if (path === '/api/experiments' && request.method === 'POST') {
        const result = await experiments.handleCreateExperiment(request, env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/experiments\/[\w-]+\/claim$/) && request.method === 'POST') {
        const experimentId = path.split('/')[3];
        const result = await experiments.handleClaimExperiment(request, env, experimentId);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/experiments\/[\w-]+\/complete$/) && request.method === 'POST') {
        const experimentId = path.split('/')[3];
        const result = await experiments.handleCompleteExperiment(request, env, experimentId);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/experiments\/[\w-]+\/fail$/) && request.method === 'POST') {
        const experimentId = path.split('/')[3];
        const result = await experiments.handleFailExperiment(request, env, experimentId);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/experiments\/[\w-]+$/) && request.method === 'GET') {
        const experimentId = path.split('/')[3];
        const result = await experiments.handleGetExperiment(experimentId, env);
        return addCors(result, corsHeaders);
      }

      // Research
      if (path === '/api/research/state' && request.method === 'GET') {
        const result = await research.handleGetResearchState(env);
        return addCors(result, corsHeaders);
      }
      if (path === '/api/research/findings' && request.method === 'GET') {
        const result = await research.handleListFindings(env);
        return addCors(result, corsHeaders);
      }
      if (path === '/api/research/findings' && request.method === 'POST') {
        const result = await research.handlePublishFinding(request, env);
        return addCors(result, corsHeaders);
      }

      // Forum
      if (path === '/api/forum/feed' && request.method === 'GET') {
        const result = await forum.handleGetFeed(request, env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/forum\/posts\/[\w-]+\/comments$/) && request.method === 'POST') {
        const postId = path.split('/')[4];
        const result = await forum.handleAddComment(request, env, postId);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/forum\/posts\/[\w-]+$/) && request.method === 'GET') {
        const postId = path.split('/')[4];
        const result = await forum.handleGetPost(postId, env);
        return addCors(result, corsHeaders);
      }
      if (path === '/api/forum/posts' && request.method === 'POST') {
        const result = await forum.handleCreatePost(request, env);
        return addCors(result, corsHeaders);
      }
      if (path === '/api/forum/search' && request.method === 'GET') {
        const result = await forum.handleSearchForum(request, env);
        return addCors(result, corsHeaders);
      }

      // Scripts
      if (path === '/api/scripts' && request.method === 'GET') {
        const result = await scripts.handleListScripts(env);
        return addCors(result, corsHeaders);
      }
      if (path === '/api/scripts' && request.method === 'POST') {
        const result = await scripts.handlePublishScript(request, env);
        return addCors(result, corsHeaders);
      }
      if (path.match(/^\/api\/scripts\/[\w-]+$/) && request.method === 'GET') {
        const scriptId = path.split('/')[3];
        const result = await scripts.handleGetScript(scriptId, env);
        return addCors(result, corsHeaders);
      }

      // Notifications
      if (path === '/api/notifications' && request.method === 'GET') {
        const result = await notifications.handleGetNotifications(request, env);
        return addCors(result, corsHeaders);
      }
      if (path === '/api/notifications/mark-read' && request.method === 'POST') {
        const result = await notifications.handleMarkRead(request, env);
        return addCors(result, corsHeaders);
      }

      // Runners
      if (path === '/api/runners' && request.method === 'GET') {
        const result = await runners.handleListRunners(env);
        return addCors(result, corsHeaders);
      }
      if (path === '/api/runners' && request.method === 'POST') {
        const result = await runners.handleRegisterRunner(request, env);
        return addCors(result, corsHeaders);
      }

      return notFound('Route');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Internal server error';
      return addCors(error(message, 500), corsHeaders);
    }
  },
};

function addCors(response: Response, headers: Record<string, string>): Response {
  const newHeaders = new Headers(response.headers);
  for (const [key, value] of Object.entries(headers)) {
    newHeaders.set(key, value);
  }
  return new Response(response.body, { status: response.status, headers: newHeaders });
}
