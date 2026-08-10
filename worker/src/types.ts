/**
 * ModelSwarm Cloudflare Worker — Type Definitions
 */

export interface Env {
  DB: D1Database;
  ENVIRONMENT: string;
  CLAIM_EXPIRY_MINUTES: string;
  STALE_THRESHOLD_MINUTES: string;
  MODELSWARM_ADMIN_KEY?: string;
}

export interface Agent {
  agent_id: string;
  name: string;
  model: string;
  role: string;
  parent_agent_id: string | null;
  api_key: string;
  registered_at: string;
  last_heartbeat: string | null;
  status: string;
  capabilities: string;
  metadata: string;
}

export interface Competition {
  competition_id: string;
  name: string;
  target: string;
  metric: string;
  higher_is_better: number;
  status: string;
  config: string;
  created_at: string;
}

export interface Experiment {
  experiment_id: string;
  hypothesis: string;
  agent_id: string;
  executing_agent_id: string | null;
  parent_agent_id: string | null;
  parent_experiment_id: string | null;
  competition_id: string | null;
  phase: number | null;
  configuration: string;
  dataset: string | null;
  features: string;
  model: string | null;
  validation_protocol: string | null;
  oof_metric: number | null;
  fold_metrics: string;
  public_score: number | null;
  runtime_seconds: number | null;
  compute_info: string;
  artifacts: string;
  decision: string | null;
  reasoning: string | null;
  status: string;
  claimed_by: string | null;
  claimed_at: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface ForumPost {
  post_id: string;
  author_id: string;
  category: string;
  title: string;
  content: string;
  experiment_id: string | null;
  tags: string;
  created_at: string;
  updated_at: string | null;
}

export interface ForumComment {
  comment_id: string;
  post_id: string;
  author_id: string;
  content: string;
  created_at: string;
}

export interface SharedScript {
  script_id: string;
  name: string;
  author_id: string;
  description: string;
  version: string;
  dependencies: string;
  usage: string | null;
  source_path: string;
  created_at: string;
  updated_at: string | null;
}

export interface Notification {
  notification_id: string;
  recipient_id: string;
  sender_id: string | null;
  type: string;
  message: string;
  link: string | null;
  created_at: string;
  read: number;
}

export interface Runner {
  runner_id: string;
  name: string;
  provider: string;
  capabilities: string;
  status: string;
  registered_at: string;
  last_heartbeat: string | null;
}
