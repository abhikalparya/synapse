import { useCallback, useId, useState, type FormEvent } from "react";
import type { ApplyResponse, Proposal } from "../types";

type Props = {
  open: boolean;
  onClose: () => void;
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

export function GenerateRoadmapModal({ open, onClose, onApplied }: Props) {
  const idBase = useId();
  const [goal, setGoal] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [proposal, setProposal] = useState<Proposal | null>(null);

  const reset = useCallback(() => {
    setGoal("");
    setBusy(false);
    setError(null);
    setProposal(null);
  }, []);

  const handleClose = useCallback(() => {
    if (busy) return;
    reset();
    onClose();
  }, [busy, onClose, reset]);

  async function handleGenerate(e: FormEvent) {
    e.preventDefault();
    const trimmed = goal.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await fetchJson<Proposal>("/generate/roadmap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: trimmed }),
      });
      setProposal(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate roadmap");
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
      reset();
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
      reset();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to discard proposal");
      setBusy(false);
    }
  }

  if (!open) return null;

  const titleById = new Map(proposal?.topics.map((t) => [t.temp_id, t.title]) ?? []);

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
            Generate roadmap
          </h2>
          <button type="button" className="modal__close" onClick={handleClose} disabled={busy} aria-label="Close">
            ×
          </button>
        </div>

        <div className="modal__body">
          {!proposal ? (
            <form onSubmit={handleGenerate}>
              <label className="modal__label" htmlFor={`${idBase}-goal`}>
                Learning goal
              </label>
              <div className="modal__goal-row">
                <textarea
                  id={`${idBase}-goal`}
                  className="modal__textarea"
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  placeholder="e.g. learn how transformers work"
                  rows={2}
                  disabled={busy}
                />
              </div>
              {error ? <p className="modal__error">{error}</p> : null}
              <div className="modal__actions">
                <button type="button" className="modal__btn modal__btn--ghost" onClick={handleClose} disabled={busy}>
                  Cancel
                </button>
                <button type="submit" className="modal__btn modal__btn--primary" disabled={busy || !goal.trim()}>
                  {busy ? "Generating…" : "Generate"}
                </button>
              </div>
            </form>
          ) : (
            <>
              <p className="review-source">Proposed from {proposal.source}. Nothing is saved until you apply.</p>

              <div className="review-section">
                <h4>Topics ({proposal.topics.length})</h4>
                <div className="review-list">
                  {proposal.topics.map((t) => (
                    <div
                      key={t.temp_id}
                      className={`review-topic${t.needs_review ? " review-topic--needs-review" : ""}`}
                    >
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

              {proposal.dependencies.length > 0 ? (
                <div className="review-section">
                  <h4>Dependencies ({proposal.dependencies.length})</h4>
                  <div className="review-list">
                    {proposal.dependencies.map((d, i) => (
                      <div className="review-dep" key={`${d.from_temp_id}-${d.to_temp_id}-${i}`}>
                        {titleById.get(d.from_temp_id) ?? d.from_temp_id}
                        <span className="review-dep__arrow">requires →</span>
                        {titleById.get(d.to_temp_id) ?? d.to_temp_id}
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
          )}
        </div>
      </div>
    </div>
  );
}
