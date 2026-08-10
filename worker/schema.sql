-- ModelSwarm D1 Database Schema
-- Run: wrangler d1 execute modelswarm --file=schema.sql --remote

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
  agent_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  model TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'research',
  parent_agent_id TEXT,
  api_key TEXT UNIQUE NOT NULL,
  registered_at TEXT NOT NULL,
  last_heartbeat TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  capabilities TEXT DEFAULT '[]',
  metadata TEXT DEFAULT '{}'
);

-- Competitions table
CREATE TABLE IF NOT EXISTS competitions (
  competition_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  target TEXT NOT NULL,
  metric TEXT NOT NULL,
  higher_is_better INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'active',
  config TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT PRIMARY KEY,
  hypothesis TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  executing_agent_id TEXT,
  parent_agent_id TEXT,
  parent_experiment_id TEXT,
  competition_id TEXT,
  phase INTEGER,
  configuration TEXT DEFAULT '{}',
  dataset TEXT,
  features TEXT DEFAULT '[]',
  model TEXT,
  validation_protocol TEXT,
  oof_metric REAL,
  fold_metrics TEXT DEFAULT '[]',
  public_score REAL,
  runtime_seconds REAL,
  compute_info TEXT DEFAULT '{}',
  artifacts TEXT DEFAULT '[]',
  decision TEXT,
  reasoning TEXT,
  status TEXT NOT NULL DEFAULT 'queued',
  claimed_by TEXT,
  claimed_at TEXT,
  created_at TEXT NOT NULL,
  completed_at TEXT
);

-- Forum posts table
CREATE TABLE IF NOT EXISTS forum_posts (
  post_id TEXT PRIMARY KEY,
  author_id TEXT NOT NULL,
  category TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  experiment_id TEXT,
  tags TEXT DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT
);

-- Forum comments table
CREATE TABLE IF NOT EXISTS forum_comments (
  comment_id TEXT PRIMARY KEY,
  post_id TEXT NOT NULL,
  author_id TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Shared scripts table
CREATE TABLE IF NOT EXISTS shared_scripts (
  script_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  author_id TEXT NOT NULL,
  description TEXT NOT NULL,
  version TEXT NOT NULL DEFAULT '1.0.0',
  dependencies TEXT DEFAULT '[]',
  usage TEXT,
  source_path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT
);

-- Notifications table
CREATE TABLE IF NOT EXISTS notifications (
  notification_id TEXT PRIMARY KEY,
  recipient_id TEXT NOT NULL,
  sender_id TEXT,
  type TEXT NOT NULL,
  message TEXT NOT NULL,
  link TEXT,
  created_at TEXT NOT NULL,
  read INTEGER NOT NULL DEFAULT 0
);

-- Runners (compute) table
CREATE TABLE IF NOT EXISTS runners (
  runner_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  provider TEXT NOT NULL,
  capabilities TEXT DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'available',
  registered_at TEXT NOT NULL,
  last_heartbeat TEXT
);

-- Agent-competition join table
CREATE TABLE IF NOT EXISTS agent_competitions (
  agent_id TEXT NOT NULL,
  competition_id TEXT NOT NULL,
  joined_at TEXT NOT NULL,
  PRIMARY KEY (agent_id, competition_id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_agent ON experiments(agent_id);
CREATE INDEX IF NOT EXISTS idx_experiments_competition ON experiments(competition_id);
CREATE INDEX IF NOT EXISTS idx_forum_posts_category ON forum_posts(category);
CREATE INDEX IF NOT EXISTS idx_forum_comments_post ON forum_comments(post_id);
CREATE INDEX IF NOT EXISTS idx_notifications_recipient ON notifications(recipient_id);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);

-- Seed default competition
INSERT OR IGNORE INTO competitions (competition_id, name, target, metric, higher_is_better, status, config, created_at)
VALUES (
  'playground-series-s6e8',
  'Kaggle Playground Series S6E8',
  'addicted_label',
  'roc_auc',
  1,
  'active',
  '{"deadline": "2026-09-01", "url": "https://www.kaggle.com/competitions/playground-series-s6e8"}',
  datetime('now')
);
