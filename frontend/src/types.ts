export type GraphNode = {
  id: string;
  group: string;
  /** Filled by react-force-graph when nodeAutoColorBy is set */
  color?: string;
  title?: string;
  summary?: string;
  key_points?: string[];
  tags?: string[];
  source_notes?: string[];
  __deg?: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
};

export type GraphLink = {
  source: string | GraphNode;
  target: string | GraphNode;
};

export type GraphData = {
  nodes: GraphNode[];
  links: GraphLink[];
};

export type QueryResponse = {
  answer: string;
  used_nodes: string[];
  updated_node: string | null;
  confidence_score: number;
  wiki_action: "updated" | "created" | "skipped";
  wiki_file: string | null;
};

export type RecentNode = {
  title: string;
  filename: string;
  created_at: string | null;
  updated_at: string | null;
  tags: string[];
};

export type StatsResponse = {
  total_nodes: number;
  total_edges: number;
  recent_nodes: RecentNode[];
  top_tags: { tag: string; count: number }[];
};

export type RefactorResponse = {
  merged_groups: number;
  pages_merged: number;
  pages_updated: number;
  pages_rewritten: number;
  errors: string[];
};
