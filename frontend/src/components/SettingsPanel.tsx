import { useEffect, useId, useState } from "react";
import type { Settings, ThinkingLevel } from "../types";

type Props = {
  open: boolean;
  onClose: () => void;
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

const THINKING_LEVELS: ThinkingLevel[] = ["standard", "extended"];

export function SettingsPanel({ open, onClose }: Props) {
  const idBase = useId();
  const [loading, setLoading] = useState(true);
  const [persona, setPersona] = useState("");
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [thinkingLevel, setThinkingLevel] = useState<ThinkingLevel>("standard");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!open) return;
    setSaved(false);
    setError(null);
    setLoading(true);
    fetchJson<Settings>("/settings")
      .then((s) => {
        setPersona(s.persona);
        setMemoryEnabled(s.memory_enabled);
        setThinkingLevel(s.thinking_level);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load settings"))
      .finally(() => setLoading(false));
  }, [open]);

  async function handleSave() {
    if (busy) return;
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      await fetchJson<Settings>("/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ persona, memory_enabled: memoryEnabled, thinking_level: thinkingLevel }),
      });
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save settings");
    } finally {
      setBusy(false);
    }
  }

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(ev) => ev.target === ev.currentTarget && onClose()}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${idBase}-title`}
        aria-busy={busy}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="modal__header">
          <h2 id={`${idBase}-title`} className="modal__title">
            Settings
          </h2>
          <button type="button" className="modal__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="modal__body">
          {loading ? (
            <p className="sidebar__muted">Loading…</p>
          ) : (
            <>
              <label className="modal__label" htmlFor={`${idBase}-persona`}>
                Persona
              </label>
              <textarea
                id={`${idBase}-persona`}
                className="modal__textarea"
                value={persona}
                onChange={(e) => setPersona(e.target.value)}
                placeholder="e.g. Answer like a terse senior engineer -- no fluff, no hedging."
                rows={3}
                disabled={busy}
              />
              <p className="review-source">
                Appended to every LLM call (ingest, expand, audit, reshape, quiz, ask) as a tone/style instruction.
              </p>

              <label className="modal__label" style={{ marginTop: "0.8rem" }}>
                <input
                  type="checkbox"
                  checked={memoryEnabled}
                  onChange={(e) => setMemoryEnabled(e.target.checked)}
                  disabled={busy}
                  style={{ marginRight: "0.4rem" }}
                />
                Remember prior questions when asking about a topic
              </label>
              <p className="review-source">
                When on, recent Q&A turns for a topic are included as context on the next question. When off, every
                question is answered fresh.
              </p>

              <label className="modal__label" htmlFor={`${idBase}-thinking`} style={{ marginTop: "0.8rem" }}>
                Thinking
              </label>
              <select
                id={`${idBase}-thinking`}
                className="resource-form__select"
                style={{ width: "100%" }}
                value={thinkingLevel}
                onChange={(e) => setThinkingLevel(e.target.value as ThinkingLevel)}
                disabled={busy}
              >
                {THINKING_LEVELS.map((t) => (
                  <option key={t} value={t}>
                    {t === "extended" ? "Extended -- reason step by step before answering" : "Standard"}
                  </option>
                ))}
              </select>

              {error ? <p className="modal__error">{error}</p> : null}
              {saved ? <p className="review-source">Saved.</p> : null}

              <div className="modal__actions">
                <button type="button" className="modal__btn modal__btn--ghost" onClick={onClose} disabled={busy}>
                  Close
                </button>
                <button type="button" className="modal__btn modal__btn--primary" onClick={() => void handleSave()} disabled={busy}>
                  {busy ? "Saving…" : "Save"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
