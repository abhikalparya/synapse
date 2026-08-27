import type { GraphNode, StatsResponse, TopicStatus } from "../types";

const STATUS_LABEL: Record<TopicStatus, string> = {
  not_started: "Not started",
  in_progress: "In progress",
  complete: "Complete",
};

type HomeProps = {
  stats: StatsResponse | null;
  loading: boolean;
  graphNodes: GraphNode[];
  onOpenTopic: (id: string) => void;
};

export function HomeWorkspace({ stats, loading, graphNodes, onOpenTopic }: HomeProps) {
  const isTopicAvailable = (id: string) => graphNodes.some((node) => node.id === id);

  return (
    <section className="workspace-view workspace-view--home" aria-labelledby="home-title">
      <header className="workspace-view__header">
        <p className="workspace-view__eyebrow">Workspace</p>
        <h1 id="home-title">What would you like to learn?</h1>
        <p>Continue with a topic or open Explore to see how your knowledge connects.</p>
      </header>

      <div className="workspace-view__summary" aria-label="Knowledge summary">
        <div>
          <span className="workspace-view__summary-value">{loading ? "—" : stats?.total_nodes ?? 0}</span>
          <span className="workspace-view__summary-label">Topics</span>
        </div>
        <div>
          <span className="workspace-view__summary-value">{loading ? "—" : stats?.total_edges ?? 0}</span>
          <span className="workspace-view__summary-label">Dependencies</span>
        </div>
      </div>

      <section className="workspace-view__section" aria-labelledby="home-recent-title">
        <div className="workspace-view__section-heading">
          <h2 id="home-recent-title">Recent topics</h2>
          <span>{loading ? "Loading…" : `${stats?.recent_nodes.length ?? 0} available`}</span>
        </div>
        {!stats?.recent_nodes.length ? (
          <p className="workspace-view__muted">
            {loading ? "Loading your topics…" : "No topics yet. Use Add knowledge to create your first learning graph."}
          </p>
        ) : (
          <div className="workspace-topic-list">
            {stats.recent_nodes.map((topic) => (
              <button
                type="button"
                className="workspace-topic-list__item"
                key={topic.id}
                onClick={() => onOpenTopic(topic.id)}
                disabled={!isTopicAvailable(topic.id)}
                title={isTopicAvailable(topic.id) ? undefined : "This topic will be available when the graph is ready."}
              >
                <span>{topic.title}</span>
                <small>{STATUS_LABEL[topic.status]}</small>
              </button>
            ))}
          </div>
        )}
      </section>
    </section>
  );
}

type LearnProps = {
  nodes: GraphNode[];
  onOpenTopic: (id: string) => void;
};

export function LearnWorkspace({ nodes, onOpenTopic }: LearnProps) {
  const sortedNodes = [...nodes].sort((a, b) => (a.title ?? a.id).localeCompare(b.title ?? b.id));

  return (
    <section className="workspace-view" aria-labelledby="learn-title">
      <header className="workspace-view__header">
        <p className="workspace-view__eyebrow">Learn</p>
        <h1 id="learn-title">Your topics</h1>
        <p>Choose a topic to open it in Explore. A fuller topic learning workspace will build on this entry point.</p>
      </header>

      {!sortedNodes.length ? (
        <p className="workspace-view__muted">No topics yet. Add knowledge to begin.</p>
      ) : (
        <div className="workspace-topic-list workspace-topic-list--grid">
          {sortedNodes.map((node) => (
            <button type="button" className="workspace-topic-list__item" key={node.id} onClick={() => onOpenTopic(node.id)}>
              <span>{node.title ?? node.id}</span>
              <small>{node.status ? STATUS_LABEL[node.status] : "Not started"}</small>
              {node.summary ? <em>{node.summary}</em> : null}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

type ReviewProps = {
  onOpenAiOperations: () => void;
};

export function ReviewWorkspace({ onOpenAiOperations }: ReviewProps) {
  return (
    <section className="workspace-view" aria-labelledby="review-title">
      <header className="workspace-view__header">
        <p className="workspace-view__eyebrow">Review</p>
        <h1 id="review-title">Review your knowledge</h1>
        <p>Quizzes, proposals, and learning evidence will gather here as this workspace grows.</p>
      </header>

      <section className="workspace-view__section workspace-view__section--quiet">
        <h2>Available now</h2>
        <p>
          Generate a proposal with Add knowledge, or select a topic in Explore to take its closure quiz. No review
          metrics are shown until there is review data to support them.
        </p>
        <button type="button" className="workspace-view__action" onClick={onOpenAiOperations}>
          Open AI operations
        </button>
      </section>
    </section>
  );
}
