/**
 * Forum API routes
 * GET  /api/forum/feed
 * GET  /api/forum/posts/:id
 * POST /api/forum/posts
 * POST /api/forum/posts/:id/comments
 * GET  /api/forum/search
 */

import { Env } from '../types';
import { ok, error, notFound, created } from '../lib/json';
import { generatePostId, generateCommentId, now } from '../lib/id';

export async function handleGetFeed(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const category = url.searchParams.get('category');
  const limit = parseInt(url.searchParams.get('limit') || '20');

  let query = 'SELECT * FROM forum_posts WHERE 1=1';
  const params: string[] = [];

  if (category) {
    query += ' AND category = ?';
    params.push(category);
  }

  query += ' ORDER BY created_at DESC LIMIT ?';
  params.push(limit.toString());

  const result = await env.DB.prepare(query).bind(...params).all();
  return ok({ posts: result.results });
}

export async function handleGetPost(postId: string, env: Env): Promise<Response> {
  const post = await env.DB
    .prepare('SELECT * FROM forum_posts WHERE post_id = ?')
    .bind(postId)
    .first();

  if (!post) return notFound('Post');

  const comments = await env.DB
    .prepare('SELECT * FROM forum_comments WHERE post_id = ? ORDER BY created_at ASC')
    .bind(postId)
    .all();

  return ok({ post, comments: comments.results });
}

export async function handleCreatePost(request: Request, env: Env): Promise<Response> {
  const body = await request.json<{ category: string; title: string; content: string; author_id?: string; experiment_id?: string; tags?: string[] }>();

  if (!body.category || !body.title || !body.content) {
    return error('category, title, and content are required');
  }

  const validCategories = ['discussion', 'proposal', 'discovery', 'announcement'];
  if (!validCategories.includes(body.category)) {
    return error(`Invalid category. Must be one of: ${validCategories.join(', ')}`);
  }

  const postId = generatePostId();
  const timestamp = now();
  const tags = JSON.stringify(body.tags || []);

  await env.DB
    .prepare(`INSERT INTO forum_posts (post_id, author_id, category, title, content, experiment_id, tags, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(postId, body.author_id || 'system', body.category, body.title, body.content, body.experiment_id || null, tags, timestamp)
    .run();

  return created({ post_id: postId, category: body.category, created_at: timestamp });
}

export async function handleAddComment(request: Request, env: Env, postId: string): Promise<Response> {
  const body = await request.json<{ content: string; author_id?: string }>();

  if (!body.content) {
    return error('content is required');
  }

  // Verify post exists
  const post = await env.DB
    .prepare('SELECT post_id FROM forum_posts WHERE post_id = ?')
    .bind(postId)
    .first();

  if (!post) return notFound('Post');

  const commentId = generateCommentId();
  const timestamp = now();

  await env.DB
    .prepare('INSERT INTO forum_comments (comment_id, post_id, author_id, content, created_at) VALUES (?, ?, ?, ?, ?)')
    .bind(commentId, postId, body.author_id || 'system', body.content, timestamp)
    .run();

  return created({ comment_id: commentId, post_id: postId, created_at: timestamp });
}

export async function handleSearchForum(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const q = url.searchParams.get('q') || '';

  if (!q) return ok({ results: [] });

  const result = await env.DB
    .prepare("SELECT * FROM forum_posts WHERE title LIKE ? OR content LIKE ? ORDER BY created_at DESC LIMIT 20")
    .bind(`%${q}%`, `%${q}%`)
    .all();

  return ok({ results: result.results });
}
