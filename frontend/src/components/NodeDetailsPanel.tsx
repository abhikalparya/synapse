import { useMemo } from "react";
import { neighborNodeIds } from "../graphUtils";
import type { GraphData, GraphNode } from "../types";

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
          <p className="details-panel__empty">Select a node on the graph to inspect wiki content.</p>
        ) : (
          <article className="node-card">
            <h3 className="node-card__title">{node.title ?? node.id}</h3>
            {node.summary ? <p className="node-card__summary">{node.summary}</p> : null}
            {node.key_points && node.key_points.length > 0 ? (
              <section className="node-card__section">
                <h4>Key points</h4>
                <ul>
                  {node.key_points.map((kp) => (
                    <li key={kp}>{kp}</li>
                  ))}
                </ul>
              </section>
            ) : null}
            {node.tags && node.tags.length > 0 ? (
              <section className="node-card__section">
                <h4>Tags</h4>
                <div className="node-card__tags">
                  {node.tags.map((t) => (
                    <span key={t} className="tag-pill">
                      {t}
                    </span>
                  ))}
                </div>
              </section>
            ) : null}
            {node.source_notes && node.source_notes.length > 0 ? (
              <section className="node-card__section">
                <h4>Source notes</h4>
                <ul className="node-card__sources">
                  {node.source_notes.map((s) => (
                    <li key={s}>{s}</li>
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
