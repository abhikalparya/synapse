import { useMemo } from "react";
import { neighborNodeIds } from "../graphUtils";
import type { GraphData, GraphNode, TopicStatus } from "../types";

const STATUS_LABEL: Record<TopicStatus, string> = {
  not_started: "Not started",
  in_progress: "In progress",
  complete: "Complete",
};

type Props = {
  graphData: GraphData;
  node: GraphNode | null;
  onClose: () => void;
  onNavigateToNode: (node: GraphNode) => void;
};

export function NodeDetailsPanel({ graphData, node, onClose, onNavigateToNode }: Props) {
  const related = useMemo(() => {
    if (!node) return [];
    const ids = neighborNodeIds(graphData, node.id);
    const byId = new Map(graphData.nodes.map((n) => [n.id, n]));
    return ids.map((id) => byId.get(id)).filter((n): n is GraphNode => Boolean(n));
  }, [graphData, node]);

  return (
    <aside className="details-panel">
      <div className="details-panel__header">
        <h2 className="details-panel__title">Connections</h2>
        {node ? (
          <button type="button" className="details-panel__close" onClick={onClose} aria-label="Close panel">
            ×
          </button>
        ) : null}
      </div>
      <div className="details-panel__body">
        {!node ? (
          <p className="details-panel__empty">Select a topic on the graph to inspect it.</p>
        ) : (
          <article className="node-card">
            <h3 className="node-card__title">{node.title ?? node.id}</h3>
            {node.status ? (
              <span className={`status-pill status-pill--${node.status}`}>
                {STATUS_LABEL[node.status]}
              </span>
            ) : null}
            {node.summary ? <p className="node-card__summary">{node.summary}</p> : null}
            {node.resources && node.resources.length > 0 ? (
              <section className="node-card__section">
                <h4>Resources</h4>
                <ul className="node-card__sources">
                  {node.resources.map((r) => (
                    <li key={r.id}>{r.title || r.source_ref}</li>
                  ))}
                </ul>
              </section>
            ) : null}
            {related.length > 0 ? (
              <section className="node-card__section node-card__section--related">
                <h4>Related concepts</h4>
                <ul className="node-card__related">
                  {related.map((n) => (
                    <li key={n.id}>
                      <button type="button" className="node-card__related-btn" onClick={() => onNavigateToNode(n)}>
                        {n.title ?? n.id}
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
          </article>
        )}
      </div>
    </aside>
  );
}
