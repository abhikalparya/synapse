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
  title?: string;
  summary?: string;
  status?: TopicStatus;
  resources?: Resource[];
  quiz_passed?: boolean;
  zone_id?: string | null;
  __deg?: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
};

export type Zone = {
  id: string;
  label: string;
  color: string | null;
  created_at: string | null;
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

export type ProposedDependencyRemoval = {
  from_topic_id: string;
  to_topic_id: string;
  reason: string;
};

export type ProposedMerge = {
  source_topic_id: string;
  target_topic_id: string;
  reason: string;
};

export type ProposedTopicEdit = {
  topic_id: string;
  new_summary: string;
  reason: string;
};

export type SkippedProposedDependency = {
  from_title: string;
  to_title: string;
  reason: string;
};

export type ProposalMode = "ingest" | "expand" | "reshape";

/** A pending/applied/discarded AI-proposed graph change, returned by one of the four
 * /ai/{ingest,expand,reshape} operation-mode endpoints. */
export type Proposal = {
  id: string;
  status: ProposalStatus;
  mode: ProposalMode;
  source: string;
  topics: ProposedTopic[];
  dependencies: ProposedDependency[];
  removed_dependencies: ProposedDependencyRemoval[];
  merges: ProposedMerge[];
  edits: ProposedTopicEdit[];
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
  removed_dependency_count: number;
  merged_topic_count: number;
  edited_topic_count: number;
  skipped_dependencies: SkippedProposedDependency[];
};

export type RollbackResponse = {
  snapshot_id: string;
  restored_topics: number;
  restored_dependencies: number;
};

export type QuizQuestionPublic = {
  id: string;
  question: string;
  choices: string[];
};

/** Quiz as returned by generation -- no correct answers included. */
export type QuizPublic = {
  topic_id: string;
  questions: QuizQuestionPublic[];
};

export type QuizResultQuestion = {
  question_id: string;
  correct: boolean;
  correct_index: number;
  selected_index: number | null;
};

export type QuizResult = {
  topic_id: string;
  score: number;
  passed: boolean;
  correct_count: number;
  total: number;
  results: QuizResultQuestion[];
};

export type AuditFindingType = "orphaned_topic" | "duplicate_title" | "thin_topic" | "missing_prerequisite" | "cycle_risk";

export type AuditFinding = {
  type: AuditFindingType;
  topic_ids: string[];
  detail: string;
};

/** Read-only diagnostic report from POST /ai/audit -- never a Proposal, no Apply/Discard. */
export type AuditReport = {
  generated_at: string;
  total_topics: number;
  findings: AuditFinding[];
  /** "partial" when structural checks ran but semantic LLM analysis was unavailable. */
  status?: "ok" | "partial";
  semantic_analysis?: "available" | "unavailable";
  semantic_error?: string | null;
  structural_findings?: AuditFinding[];
};

export type ArtifactType = "note" | "code_snippet" | "summary" | "generated_output" | "qa_log";

/** Something the learner PRODUCED while studying a topic -- distinct from Resource
 * (something they studied FROM). */
export type Artifact = {
  id: string;
  topic_id: string;
  type: ArtifactType;
  title: string;
  content: string;
  created_at: string | null;
};

/** One question/answer turn with the in-session assistant, scoped to a single topic. */
export type AskResponse = {
  answer: string;
  artifact_id: string;
};

export type ThinkingLevel = "standard" | "extended";

/** Local workspace-level LLM behavior settings (Phase 13) -- singleton, no per-user concept. */
export type Settings = {
  persona: string;
  memory_enabled: boolean;
  thinking_level: ThinkingLevel;
};
