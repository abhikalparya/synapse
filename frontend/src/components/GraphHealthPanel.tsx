import type { GraphNode, LintIssue, LintIssueType } from "../types";

type Props = {
  issues: LintIssue[];
  loading: boolean;
  error: string | null;
  nodes: GraphNode[];
  onOpenTopic: (id: string) => void;
  onRetry: () => void;
};

const TYPE_LABEL: Record<LintIssueType, string> = {
  missing_summary: "Missing summary",
  duplicate_title: "Duplicate title",
  self_dependency: "Self dependency",
  orphan_dependency: "Orphan dependency",
  cycle: "Cycle",
};

// Render order -- structural DAG problems first, then content-quality issues.
const TYPE_ORDER: LintIssueType[] = ["cycle", "self_dependency", "orphan_dependency", "duplicate_title", "missing_summary"];

function titleFor(nodes: GraphNode[], id: string): string {
  return nodes.find((n) => n.id === id)?.title ?? id;
}

function nodesMatchingTitle(nodes: GraphNode[], title: string): GraphNode[] {
  const key = title.trim().toLowerCase();
  return nodes.filter((n) => (n.title ?? "").trim().toLowerCase() === key);
}

function TopicLink({ id, nodes, onOpenTopic }: { id: string; nodes: GraphNode[]; onOpenTopic: (id: string) => void }) {
  return (
    <button type="button" className="health-finding__link" onClick={() => onOpenTopic(id)}>
      {titleFor(nodes, id)}
    </button>
  );
}

function FindingBody({ issue, nodes, onOpenTopic }: { issue: LintIssue; nodes: GraphNode[]; onOpenTopic: (id: string) => void }) {
  switch (issue.type) {
    case "missing_summary":
    case "self_dependency":
      return issue.topic ? (
        <p className="audit-finding__detail">
          <TopicLink id={issue.topic} nodes={nodes} onOpenTopic={onOpenTopic} />
        </p>
      ) : (
        <p className="audit-finding__detail">{issue.detail ?? "Unknown topic"}</p>
      );

    case "cycle": {
      const chain = issue.topics ?? [];
      if (chain.length === 0) return <p className="audit-finding__detail">{issue.detail ?? "Cycle detected"}</p>;
      return (
        <p className="audit-finding__detail health-finding__chain">
          {chain.map((id, i) => (
            <span key={`${id}-${i}`}>
              <TopicLink id={id} nodes={nodes} onOpenTopic={onOpenTopic} />
              {i < chain.length - 1 ? <span className="health-finding__arrow"> → </span> : null}
            </span>
          ))}
        </p>
      );
    }

    case "duplicate_title": {
      const titles = issue.topics ?? [];
      return (
        <div className="audit-finding__detail health-finding__duplicates">
          {titles.map((title, i) => {
            const matches = nodesMatchingTitle(nodes, title);
            return (
              <p key={`${title}-${i}`} className="health-finding__duplicate-row">
                <span>{title}</span>
                {matches.length > 0 ? (
                  <span className="health-finding__duplicate-links">
                    {matches.map((n) => (
                      <TopicLink key={n.id} id={n.id} nodes={nodes} onOpenTopic={onOpenTopic} />
                    ))}
                  </span>
                ) : null}
              </p>
            );
          })}
        </div>
      );
    }

    case "orphan_dependency":
    default:
      return <p className="audit-finding__detail">{issue.detail ?? "Unresolvable reference"}</p>;
  }
}

export function GraphHealthPanel({ issues, loading, error, nodes, onOpenTopic, onRetry }: Props) {
  if (loading) {
    return (
      <div className="app__canvas-loading" aria-live="polite">
        <span className="app__canvas-loading__dot" aria-hidden />
        Checking graph health…
      </div>
    );
  }

  if (error) {
    return (
      <div className="app__error" role="alert">
        <p>{error}</p>
        <button type="button" className="workspace-view__action" onClick={onRetry}>
          Retry
        </button>
      </div>
    );
  }

  if (issues.length === 0) {
    return <p className="sidebar__muted">No issues found. Your graph looks structurally sound.</p>;
  }

  const grouped = TYPE_ORDER.map((type) => ({ type, items: issues.filter((i) => i.type === type) })).filter(
    (g) => g.items.length > 0,
  );

  return (
    <div className="health-panel">
      <p className="review-source">
        {issues.length} finding{issues.length === 1 ? "" : "s"} across {grouped.length} categor
        {grouped.length === 1 ? "y" : "ies"}.
      </p>
      {grouped.map(({ type, items }) => (
        <div className="health-panel__group" key={type}>
          <h3 className="health-panel__group-title">
            {TYPE_LABEL[type]} ({items.length})
          </h3>
          <div className="health-panel__list">
            {items.map((issue, i) => (
              <div className={`audit-finding audit-finding--${issue.type}`} key={i}>
                <span className="audit-finding__type">{TYPE_LABEL[issue.type]}</span>
                <FindingBody issue={issue} nodes={nodes} onOpenTopic={onOpenTopic} />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
