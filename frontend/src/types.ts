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
