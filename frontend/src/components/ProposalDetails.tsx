import type { GraphNode, Proposal } from "../types";

type Props = {
  proposal: Proposal;
  nodes: GraphNode[];
  busy?: boolean;
  error?: string | null;
  onApply?: () => void;
  onDiscard?: () => void;
};

const MODE_LABEL: Record<Proposal["mode"], string> = {
  ingest: "Ingest",
  expand: "Expand",
  reshape: "Reshape",
};

function confidenceTier(confidence: number): "high" | "medium" | "low" {
  if (confidence >= 0.8) return "high";
  if (confidence >= 0.6) return "medium";
  return "low";
}

export function ProposalDetails({ proposal, nodes, busy = false, error, onApply, onDiscard }: Props) {
  const titleById = new Map<string, string>([
    ...proposal.topics.map((topic) => [topic.temp_id, topic.title] as const),
    ...nodes.map((node) => [node.id, node.title ?? node.id] as const),
  ]);
  const resolveTitle = (id: string) => titleById.get(id) ?? id;

  return (
    <>
      <p className="review-source">
        {MODE_LABEL[proposal.mode]} proposal from {proposal.source}. Nothing is saved until you apply.
      </p>

      {proposal.topics.length > 0 ? (
        <div className="review-section">
          <h4>New topics ({proposal.topics.length})</h4>
          <div className="review-list">
            {proposal.topics.map((topic) => (
              <div key={topic.temp_id} className={`review-topic${topic.needs_review ? " review-topic--needs-review" : ""}`}>
                <div>
                  <p className="review-topic__title">{topic.title}</p>
                  {topic.summary ? <p className="review-topic__summary">{topic.summary}</p> : null}
                </div>
                <span className={`confidence-badge confidence-badge--${confidenceTier(topic.confidence)}`}>
                  {topic.needs_review ? "needs review · " : ""}
                  {Math.round(topic.confidence * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {proposal.dependencies.length > 0 ? (
        <div className="review-section">
          <h4>New dependencies ({proposal.dependencies.length})</h4>
          <div className="review-list">
            {proposal.dependencies.map((dependency, index) => (
              <div className="review-dep" key={`${dependency.from_temp_id}-${dependency.to_temp_id}-${index}`}>
                {resolveTitle(dependency.from_temp_id)}
                <span className="review-dep__arrow">requires →</span>
                {resolveTitle(dependency.to_temp_id)}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {proposal.removed_dependencies.length > 0 ? (
        <div className="review-section">
          <h4>Removed dependencies ({proposal.removed_dependencies.length})</h4>
          <div className="review-list">
            {proposal.removed_dependencies.map((dependency, index) => (
              <div className="review-dep" key={`${dependency.from_topic_id}-${dependency.to_topic_id}-${index}`}>
                {resolveTitle(dependency.from_topic_id)}
                <span className="review-dep__arrow">no longer requires →</span>
                {resolveTitle(dependency.to_topic_id)}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {proposal.merges.length > 0 ? (
        <div className="review-section">
          <h4>Merges ({proposal.merges.length})</h4>
          <div className="review-list">
            {proposal.merges.map((merge, index) => (
              <div className="review-dep" key={`${merge.source_topic_id}-${merge.target_topic_id}-${index}`}>
                {resolveTitle(merge.source_topic_id)}
                <span className="review-dep__arrow">merges into →</span>
                {resolveTitle(merge.target_topic_id)}
                {merge.reason ? <span className="review-dep__reason"> -- {merge.reason}</span> : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {proposal.edits.length > 0 ? (
        <div className="review-section">
          <h4>Edits ({proposal.edits.length})</h4>
          <div className="review-list">
            {proposal.edits.map((edit, index) => (
              <div className="review-topic" key={`${edit.topic_id}-${index}`}>
                <div>
                  <p className="review-topic__title">{resolveTitle(edit.topic_id)}</p>
                  <p className="review-topic__summary">{edit.new_summary}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {proposal.skipped_dependencies.length > 0 ? (
        <div className="review-section">
          <h4>Skipped dependencies ({proposal.skipped_dependencies.length})</h4>
          <div className="review-list">
            {proposal.skipped_dependencies.map((skipped, index) => (
              <div className="review-skipped" key={`${skipped.from_title}-${skipped.to_title}-${index}`}>
                {skipped.from_title} → {skipped.to_title}: {skipped.reason}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {proposal.errors.length > 0 ? (
        <div className="review-section">
          <h4>Errors ({proposal.errors.length})</h4>
          <div className="review-list">
            {proposal.errors.map((message, index) => (
              <div className="review-skipped" key={index}>
                {message}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {error ? <p className="modal__error">{error}</p> : null}

      {onApply || onDiscard ? (
        <div className="modal__actions">
          {onDiscard ? (
            <button type="button" className="modal__btn modal__btn--ghost" onClick={onDiscard} disabled={busy}>
              Discard
            </button>
          ) : null}
          {onApply ? (
            <button type="button" className="modal__btn modal__btn--primary" onClick={onApply} disabled={busy}>
              {busy ? "Applying…" : "Apply"}
            </button>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
