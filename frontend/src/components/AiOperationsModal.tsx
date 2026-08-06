import { useCallback, useId, useState, type FormEvent } from "react";
import type { ApplyResponse, AuditReport, GraphNode, Proposal } from "../types";

type Mode = "ingest" | "expand" | "audit" | "reshape";

const MODE_LABEL: Record<Mode, string> = {
  ingest: "Ingest",
  expand: "Expand",
  audit: "Audit",
  reshape: "Reshape",
};

type Props = {
  open: boolean;
  onClose: () => void;
  nodes: GraphNode[];
  /** Called after a successful apply so the caller can refresh the graph/stats. */
  onApplied: (result: ApplyResponse) => void;
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

function confidenceTier(confidence: number): "high" | "medium" | "low" {
  if (confidence >= 0.8) return "high";
  if (confidence >= 0.6) return "medium";
  return "low";
}

export function AiOperationsModal({ open, onClose, nodes, onApplied }: Props) {
  const idBase = useId();
  const [mode, setMode] = useState<Mode>("ingest");

  const [goal, setGoal] = useState("");
  const [expandTopicId, setExpandTopicId] = useState("");
  const [expandInstructions, setExpandInstructions] = useState("");
  const [reshapeTopicIds, setReshapeTopicIds] = useState<Set<string>>(new Set());
  const [reshapeInstructions, setReshapeInstructions] = useState("");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [auditReport, setAuditReport] = useState<AuditReport | null>(null);

  const resetAll = useCallback(() => {
    setMode("ingest");
    setGoal("");
    setExpandTopicId("");
    setExpandInstructions("");
    setReshapeTopicIds(new Set());
    setReshapeInstructions("");
    setBusy(false);
    setError(null);
    setProposal(null);
    setAuditReport(null);
  }, []);

  const handleClose = useCallback(() => {
    if (busy) return;
    resetAll();
    onClose();
  }, [busy, onClose, resetAll]);

  const switchMode = useCallback(
    (next: Mode) => {
      if (busy) return;
      setMode(next);
      setError(null);
      setProposal(null);
      setAuditReport(null);
    },
    [busy],
  );

  async function handleIngest(e: FormEvent) {
    e.preventDefault();
    const trimmed = goal.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await fetchJson<Proposal>("/ai/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: trimmed }),
      });
      setProposal(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run ingest");
    } finally {
      setBusy(false);
    }
  }

  async function handleExpand(e: FormEvent) {
    e.preventDefault();
    if (!expandTopicId || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await fetchJson<Proposal>("/ai/expand", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic_id: expandTopicId, instructions: expandInstructions.trim() || null }),
      });
      setProposal(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run expand");
    } finally {
      setBusy(false);
    }
  }

  async function handleReshape(e: FormEvent) {
    e.preventDefault();
    if (reshapeTopicIds.size === 0 || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await fetchJson<Proposal>("/ai/reshape", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic_ids: [...reshapeTopicIds],
          instructions: reshapeInstructions.trim() || null,
        }),
      });
      setProposal(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run reshape");
    } finally {
      setBusy(false);
    }
  }

  async function handleAudit() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await fetchJson<AuditReport>("/ai/audit", { method: "POST" });
      setAuditReport(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run audit");
    } finally {
      setBusy(false);
    }
  }

  async function handleApply() {
    if (!proposal || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await fetchJson<ApplyResponse>("/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ proposal_id: proposal.id }),
      });
      onApplied(result);
      resetAll();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to apply proposal");
      setBusy(false);
    }
  }

  async function handleDiscard() {
    if (!proposal || busy) return;
    setBusy(true);
    setError(null);
    try {
      await fetchJson("/discard", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ proposal_id: proposal.id }),
      });
      setProposal(null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to discard proposal");
    } finally {
      setBusy(false);
    }
  }

  function toggleReshapeTopic(id: string) {
    setReshapeTopicIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  if (!open) return null;

  const titleById = new Map<string, string>([
    ...(proposal?.topics.map((t) => [t.temp_id, t.title] as const) ?? []),
    ...nodes.map((n) => [n.id, n.title ?? n.id] as const),
  ]);
  const resolveTitle = (id: string) => titleById.get(id) ?? id;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(ev) => ev.target === ev.currentTarget && handleClose()}>
      <div
        className="modal modal--review"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${idBase}-title`}
        aria-busy={busy}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="modal__header">
          <h2 id={`${idBase}-title`} className="modal__title">
            AI operations
          </h2>
          <button type="button" className="modal__close" onClick={handleClose} disabled={busy} aria-label="Close">
            ×
          </button>
        </div>

        <div className="modal__body">
          <div className="mode-tabs" role="tablist">
            {(Object.keys(MODE_LABEL) as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                role="tab"
                aria-selected={mode === m}
                className={`mode-tab${mode === m ? " mode-tab--active" : ""}`}
                onClick={() => switchMode(m)}
                disabled={busy}
              >
                {MODE_LABEL[m]}
              </button>
            ))}
          </div>

          {mode === "ingest" && !proposal ? (
            <form onSubmit={handleIngest}>
              <label className="modal__label" htmlFor={`${idBase}-goal`}>
                Learning goal
              </label>
              <textarea
                id={`${idBase}-goal`}
                className="modal__textarea"
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="e.g. learn how transformers work"
                rows={2}
                disabled={busy}
              />
              {error ? <p className="modal__error">{error}</p> : null}
              <div className="modal__actions">
                <button type="button" className="modal__btn modal__btn--ghost" onClick={handleClose} disabled={busy}>
                  Cancel
                </button>
                <button type="submit" className="modal__btn modal__btn--primary" disabled={busy || !goal.trim()}>
                  {busy ? "Running…" : "Ingest"}
                </button>
              </div>
            </form>
          ) : null}

          {mode === "expand" && !proposal ? (
            <form onSubmit={handleExpand}>
              <label className="modal__label" htmlFor={`${idBase}-expand-topic`}>
                Topic to expand
              </label>
              <select
                id={`${idBase}-expand-topic`}
                className="resource-form__select"
                style={{ width: "100%" }}
                value={expandTopicId}
                onChange={(e) => setExpandTopicId(e.target.value)}
                disabled={busy}
              >
                <option value="">Select a topic…</option>
                {nodes.map((n) => (
                  <option key={n.id} value={n.id}>
                    {n.title ?? n.id}
                  </option>
                ))}
              </select>
              <label className="modal__label" htmlFor={`${idBase}-expand-instructions`}>
                Instructions (optional)
              </label>
              <textarea
                id={`${idBase}-expand-instructions`}
                className="modal__textarea"
                value={expandInstructions}
                onChange={(e) => setExpandInstructions(e.target.value)}
                placeholder="e.g. focus on the encoder side"
                rows={2}
                disabled={busy}
              />
              {error ? <p className="modal__error">{error}</p> : null}
              <div className="modal__actions">
                <button type="button" className="modal__btn modal__btn--ghost" onClick={handleClose} disabled={busy}>
                  Cancel
                </button>
                <button type="submit" className="modal__btn modal__btn--primary" disabled={busy || !expandTopicId}>
                  {busy ? "Running…" : "Expand"}
                </button>
              </div>
            </form>
          ) : null}

          {mode === "reshape" && !proposal ? (
            <form onSubmit={handleReshape}>
              <label className="modal__label">Topics to reshape</label>
              <div className="topic-checklist">
                {nodes.length === 0 ? <p className="sidebar__muted">No topics yet.</p> : null}
                {nodes.map((n) => (
                  <label key={n.id} className="topic-checklist__item">
                    <input
                      type="checkbox"
                      checked={reshapeTopicIds.has(n.id)}
                      onChange={() => toggleReshapeTopic(n.id)}
                      disabled={busy}
                    />
                    {n.title ?? n.id}
                  </label>
                ))}
              </div>
              <label className="modal__label" htmlFor={`${idBase}-reshape-instructions`}>
                Instructions (optional)
              </label>
              <textarea
                id={`${idBase}-reshape-instructions`}
                className="modal__textarea"
                value={reshapeInstructions}
                onChange={(e) => setReshapeInstructions(e.target.value)}
                placeholder="e.g. merge these near-duplicate topics"
                rows={2}
                disabled={busy}
              />
              {error ? <p className="modal__error">{error}</p> : null}
              <div className="modal__actions">
                <button type="button" className="modal__btn modal__btn--ghost" onClick={handleClose} disabled={busy}>
                  Cancel
                </button>
                <button type="submit" className="modal__btn modal__btn--primary" disabled={busy || reshapeTopicIds.size === 0}>
                  {busy ? "Running…" : "Reshape"}
                </button>
              </div>
            </form>
          ) : null}

          {mode === "audit" && !auditReport ? (
            <div>
              <p className="review-source">Read-only analysis of the current graph -- never mutates anything.</p>
              {error ? <p className="modal__error">{error}</p> : null}
              <div className="modal__actions">
                <button type="button" className="modal__btn modal__btn--ghost" onClick={handleClose} disabled={busy}>
                  Cancel
                </button>
                <button type="button" className="modal__btn modal__btn--primary" onClick={() => void handleAudit()} disabled={busy}>
                  {busy ? "Running…" : "Run audit"}
                </button>
              </div>
            </div>
          ) : null}

          {auditReport ? (
            <div>
              <p className="review-source">
                {auditReport.total_topics} topic(s) analyzed, {auditReport.findings.length} finding(s).
              </p>
              {auditReport.findings.length === 0 ? (
                <p className="sidebar__muted">No issues found.</p>
              ) : (
                <div className="review-list">
                  {auditReport.findings.map((f, i) => (
                    <div className={`audit-finding audit-finding--${f.type}`} key={i}>
                      <span className="audit-finding__type">{f.type.replace(/_/g, " ")}</span>
                      <p className="audit-finding__detail">{f.detail}</p>
                      {f.topic_ids.length > 0 ? (
                        <p className="audit-finding__topics">{f.topic_ids.map(resolveTitle).join(", ")}</p>
                      ) : null}
                    </div>
                  ))}
                </div>
              )}
              <div className="modal__actions">
                <button type="button" className="modal__btn modal__btn--primary" onClick={handleClose}>
                  Close
                </button>
              </div>
            </div>
          ) : null}

          {proposal ? (
            <>
              <p className="review-source">
                {MODE_LABEL[proposal.mode]} proposal from {proposal.source}. Nothing is saved until you apply.
              </p>

              {proposal.topics.length > 0 ? (
                <div className="review-section">
                  <h4>New topics ({proposal.topics.length})</h4>
                  <div className="review-list">
                    {proposal.topics.map((t) => (
                      <div key={t.temp_id} className={`review-topic${t.needs_review ? " review-topic--needs-review" : ""}`}>
                        <div>
                          <p className="review-topic__title">{t.title}</p>
                          {t.summary ? <p className="review-topic__summary">{t.summary}</p> : null}
                        </div>
                        <span className={`confidence-badge confidence-badge--${confidenceTier(t.confidence)}`}>
                          {t.needs_review ? "needs review · " : ""}
                          {Math.round(t.confidence * 100)}%
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
                    {proposal.dependencies.map((d, i) => (
                      <div className="review-dep" key={`${d.from_temp_id}-${d.to_temp_id}-${i}`}>
                        {resolveTitle(d.from_temp_id)}
                        <span className="review-dep__arrow">requires →</span>
                        {resolveTitle(d.to_temp_id)}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {proposal.removed_dependencies.length > 0 ? (
                <div className="review-section">
                  <h4>Removed dependencies ({proposal.removed_dependencies.length})</h4>
                  <div className="review-list">
                    {proposal.removed_dependencies.map((d, i) => (
                      <div className="review-dep" key={`${d.from_topic_id}-${d.to_topic_id}-${i}`}>
                        {resolveTitle(d.from_topic_id)}
                        <span className="review-dep__arrow">no longer requires →</span>
                        {resolveTitle(d.to_topic_id)}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {proposal.merges.length > 0 ? (
                <div className="review-section">
                  <h4>Merges ({proposal.merges.length})</h4>
                  <div className="review-list">
                    {proposal.merges.map((m, i) => (
                      <div className="review-dep" key={`${m.source_topic_id}-${m.target_topic_id}-${i}`}>
                        {resolveTitle(m.source_topic_id)}
                        <span className="review-dep__arrow">merges into →</span>
                        {resolveTitle(m.target_topic_id)}
                        {m.reason ? <span className="review-dep__reason"> -- {m.reason}</span> : null}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {proposal.edits.length > 0 ? (
                <div className="review-section">
                  <h4>Edits ({proposal.edits.length})</h4>
                  <div className="review-list">
                    {proposal.edits.map((e, i) => (
                      <div className="review-topic" key={`${e.topic_id}-${i}`}>
                        <div>
                          <p className="review-topic__title">{resolveTitle(e.topic_id)}</p>
                          <p className="review-topic__summary">{e.new_summary}</p>
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
                    {proposal.skipped_dependencies.map((s, i) => (
                      <div className="review-skipped" key={`${s.from_title}-${s.to_title}-${i}`}>
                        {s.from_title} → {s.to_title}: {s.reason}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {proposal.errors.length > 0 ? (
                <div className="review-section">
                  <h4>Errors ({proposal.errors.length})</h4>
                  <div className="review-list">
                    {proposal.errors.map((e, i) => (
                      <div className="review-skipped" key={i}>
                        {e}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {error ? <p className="modal__error">{error}</p> : null}

              <div className="modal__actions">
                <button type="button" className="modal__btn modal__btn--ghost" onClick={handleDiscard} disabled={busy}>
                  Discard
                </button>
                <button type="button" className="modal__btn modal__btn--primary" onClick={handleApply} disabled={busy}>
                  {busy ? "Applying…" : "Apply"}
                </button>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
