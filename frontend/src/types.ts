export type TopicStatus = "not_started" | "in_progress" | "complete";

export type Resource = {
  id: string;
  type: "link" | "document" | "note";
  source_ref: string;
  title: string;
};

export type GraphNode = {
  id: string;
  group: string;
  /** Filled by react-force-graph when nodeAutoColorBy is set */
  color?: string;
  title?: string;
  summary?: string;
  status?: TopicStatus;
  resources?: Resource[];
  __deg?: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
};

/** Directed prerequisite edge: ``source`` requires ``target``. */
export type GraphLink = {
  source: string | GraphNode;
  target: string | GraphNode;
};

export type GraphData = {
  nodes: GraphNode[];
  links: GraphLink[];
};

export type RecentNode = {
  id: string;
  title: string;
  status: TopicStatus;
  created_at: string | null;
  updated_at: string | null;
};

export type StatsResponse = {
  total_nodes: number;
  total_edges: number;
  recent_nodes: RecentNode[];
};

export type PathChainEntry = {
  id: string;
  title: string;
  status: TopicStatus;
};

export type PathEdge = {
  source: string;
  target: string;
};

/** Ordered prerequisite chain leading to ``target`` (root topics first, target last). */
export type PathResponse = {
  target: string;
  chain: PathChainEntry[];
  edges: PathEdge[];
};

export type ProposalStatus = "pending" | "applied" | "discarded";

export type ProposedTopic = {
  temp_id: string;
  title: string;
  summary: string;
  confidence: number;
  needs_review: boolean;
};

export type ProposedDependency = {
  from_temp_id: string;
  to_temp_id: string;
};

export type SkippedProposedDependency = {
  from_title: string;
  to_title: string;
  reason: string;
};

/** A pending/applied/discarded AI-proposed graph change, returned by POST /generate/roadmap. */
export type Proposal = {
  id: string;
  status: ProposalStatus;
  source: string;
  topics: ProposedTopic[];
  dependencies: ProposedDependency[];
  skipped_dependencies: SkippedProposedDependency[];
  errors: string[];
  created_at: string | null;
  applied_at: string | null;
  snapshot_id: string | null;
};

export type Topic = {
  id: string;
  title: string;
  summary: string;
  status: TopicStatus;
  resources: Resource[];
  created_at: string | null;
  updated_at: string | null;
};

export type ApplyResponse = {
  proposal_id: string;
  snapshot_id: string;
  created_topics: Topic[];
  created_dependencies: { id: string; from_topic_id: string; to_topic_id: string }[];
  skipped_dependencies: SkippedProposedDependency[];
};

export type RollbackResponse = {
  snapshot_id: string;
  restored_topics: number;
  restored_dependencies: number;
};
