import { useEffect, useMemo, useRef, useState } from "react";
import { relationshipSets } from "../graphUtils";
import type { Dependency, GraphNode, TopicStatus, Zone } from "../types";

const STATUS_LABEL: Record<TopicStatus, string> = {
  not_started: "Not started",
  in_progress: "In progress",
  complete: "Complete",
};

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const text = await res.text();
  if (!res.ok) {
    let message = text || res.statusText;
    try {
      const j = JSON.parse(text) as { detail?: unknown };
      if (typeof j.detail === "string") message = j.detail;
    } catch {
      // not JSON -- fall back to raw text
    }
    throw new Error(message);
  }
  return text ? (JSON.parse(text) as T) : (undefined as T);
}

type Props = {
  nodes: GraphNode[];
  dependencies: Dependency[];
  dependenciesLoading: boolean;
  dependenciesError: string | null;
  node: GraphNode | null;
  onClose: () => void;
  onNavigateToNode: (node: GraphNode) => void;
  onOpenInLearn: (id: string) => void;
  onTopicChanged: () => void;
  zones: Zone[];
};

export function NodeDetailsPanel({
  nodes,
  dependencies,
  dependenciesLoading,
  dependenciesError,
  node,
  onClose,
  onNavigateToNode,
  onOpenInLearn,
  onTopicChanged,
  zones,
}: Props) {
  const { prerequisites, dependents } = useMemo(() => {
    if (!node) return { prerequisites: [], dependents: [] };
    const relationships = relationshipSets(
      dependencies.map((dependency) => ({
        source: dependency.from_topic_id,
        target: dependency.to_topic_id,
      })),
      node.id,
    );
    const byId = new Map(nodes.map((candidate) => [candidate.id, candidate]));
    const resolve = (ids: Set<string>) =>
      [...ids].map((id) => byId.get(id)).filter((candidate): candidate is GraphNode => Boolean(candidate));
    return {
      prerequisites: resolve(relationships.prerequisiteIds),
      dependents: resolve(relationships.dependentIds),
    };
  }, [dependencies, node, nodes]);
  const [zoneBusy, setZoneBusy] = useState(false);
  const [zoneError, setZoneError] = useState<string | null>(null);
  const mountedRef = useRef(false);
  const activeNodeIdRef = useRef(node?.id);
  activeNodeIdRef.current = node?.id;

  useEffect(() => {
    setZoneError(null);
    setZoneBusy(false);
  }, [node?.id]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  async function handleSetZone(zoneId: string) {
    if (!node || zoneBusy) return;
    const requestNodeId = node.id;
    setZoneBusy(true);
    setZoneError(null);
    try {
      await fetchJson(`/topics/${encodeURIComponent(requestNodeId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ zone_id: zoneId || null }),
      });
      if (mountedRef.current && activeNodeIdRef.current === requestNodeId) {
        onTopicChanged();
      }
    } catch (err) {
      if (mountedRef.current && activeNodeIdRef.current === requestNodeId) {
        setZoneError(err instanceof Error ? err.message : "Failed to update zone");
      }
    } finally {
      if (mountedRef.current && activeNodeIdRef.current === requestNodeId) {
        setZoneBusy(false);
      }
    }
  }

  return (
    <aside className="details-panel">
      <div className="details-panel__header">
        <h2 className="details-panel__title">Graph context</h2>
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
            <span className={`status-pill status-pill--${node.status ?? "not_started"}`}>
              {STATUS_LABEL[node.status ?? "not_started"]}
            </span>
            {node.summary ? <p className="node-card__summary">{node.summary}</p> : null}

            <div className="details-panel__actions">
              <button type="button" className="workspace-view__action" onClick={() => onOpenInLearn(node.id)}>
                Open in Learn
              </button>
            </div>

            <section className="node-card__section">
              <h4>Zone</h4>
              <select
                className="resource-form__select"
                style={{ width: "100%" }}
                value={node.zone_id ?? ""}
                onChange={(event) => void handleSetZone(event.target.value)}
                disabled={zoneBusy}
              >
                <option value="">No zone</option>
                {zones.map((zone) => (
                  <option key={zone.id} value={zone.id}>
                    {zone.label}
                  </option>
                ))}
              </select>
              {zoneError ? <p className="modal__error">{zoneError}</p> : null}
            </section>

            <section className="node-card__section node-card__section--related">
              <h4>Prerequisites</h4>
              {dependenciesLoading ? (
                <p className="sidebar__muted">Loading relationships…</p>
              ) : dependenciesError ? (
                <p className="modal__error">{dependenciesError}</p>
              ) : prerequisites.length ? (
                <ul className="node-card__related">
                  {prerequisites.map((relatedNode) => (
                    <li key={relatedNode.id}>
                      <button type="button" className="node-card__related-btn" onClick={() => onNavigateToNode(relatedNode)}>
                        {relatedNode.title ?? relatedNode.id}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="sidebar__muted">No prerequisites.</p>
              )}
            </section>

            <section className="node-card__section node-card__section--related">
              <h4>Dependents</h4>
              {dependenciesLoading ? (
                <p className="sidebar__muted">Loading relationships…</p>
              ) : dependenciesError ? (
                <p className="modal__error">{dependenciesError}</p>
              ) : dependents.length ? (
                <ul className="node-card__related">
                  {dependents.map((relatedNode) => (
                    <li key={relatedNode.id}>
                      <button type="button" className="node-card__related-btn" onClick={() => onNavigateToNode(relatedNode)}>
                        {relatedNode.title ?? relatedNode.id}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="sidebar__muted">No dependents.</p>
              )}
            </section>
          </article>
        )}
      </div>
    </aside>
  );
}
