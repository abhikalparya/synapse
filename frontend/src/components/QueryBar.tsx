import { useEffect, useState, type FormEvent } from "react";
import { renderAnswerMarkdown } from "../renderAnswerMarkdown";

const LOADING_PHRASES = [
  "Analyzing knowledge…",
  "Connecting concepts…",
  "Traversing your graph…",
  "Synthesizing insight…",
];

type Props = {
  answer: string | null;
  loading: boolean;
  error: string | null;
  answerEpoch: number;
  queryHistory?: readonly string[];
  onReplayHistory?: (query: string) => void;
  onSubmit: (text: string) => void | Promise<void>;
  onDismissError?: () => void;
};

export function QueryBar({
  answer,
  loading,
  error,
  answerEpoch,
  queryHistory = [],
  onReplayHistory,
  onSubmit,
  onDismissError,
}: Props) {
  const [value, setValue] = useState("");
  const [loadingPhraseIdx, setLoadingPhraseIdx] = useState(0);

  useEffect(() => {
    if (!loading) {
      setLoadingPhraseIdx(0);
      return;
    }
    const id = window.setInterval(() => {
      setLoadingPhraseIdx((i) => (i + 1) % LOADING_PHRASES.length);
    }, 2200);
    return () => window.clearInterval(id);
  }, [loading]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const q = value.trim();
    if (!q || loading) return;
    try {
      await onSubmit(q);
      setValue("");
    } catch {
      /* keep draft on failure */
    }
  }

  const showPanel = Boolean(answer || loading || error);
  const loadingLabel = LOADING_PHRASES[loadingPhraseIdx] ?? LOADING_PHRASES[0];

  return (
    <div className={`query-bar${loading ? " query-bar--busy" : ""}`}>
      {queryHistory.length > 0 ? (
        <div className="query-bar__history" aria-label="Recent queries">
          {queryHistory.map((q) => (
            <button
              key={q}
              type="button"
              className="query-bar__history-chip"
              disabled={loading}
              onClick={() => onReplayHistory?.(q)}
              title="Re-run query"
            >
              {q.length > 42 ? `${q.slice(0, 40)}…` : q}
            </button>
          ))}
        </div>
      ) : null}

      {showPanel ? (
        <div className="query-bar__panel">
          {error ? (
            <div className="query-bar__error" role="alert">
              <div className="query-bar__error-body">
                <span className="query-bar__error-title">Something went wrong</span>
                <p className="query-bar__error-text">{error}</p>
              </div>
              {onDismissError ? (
                <button
                  type="button"
                  className="query-bar__error-dismiss"
                  onClick={onDismissError}
                  aria-label="Dismiss error"
                >
                  ×
                </button>
              ) : null}
            </div>
          ) : null}

          {loading ? (
            <div className="query-bar__thinking" aria-live="polite" aria-busy="true">
              <span className="query-bar__thinking-dots" aria-hidden>
                <span />
                <span />
                <span />
              </span>
              <span className="query-bar__thinking-label">{loadingLabel}</span>
            </div>
          ) : null}

          {answer ? (
            <div
              className={`query-bar__answer${loading ? " query-bar__answer--stale" : ""}`}
              key={answerEpoch}
            >
              <div className="query-bar__answer-card">
                <div className="query-bar__answer-label">Answer</div>
                <div className="query-bar__answer-text query-bar__answer-text--md">{renderAnswerMarkdown(answer)}</div>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      <form className="query-bar__form" onSubmit={handleSubmit} aria-busy={loading}>
        <input
          className="query-bar__input"
          placeholder="Ask the knowledge graph…"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={loading}
          autoComplete="off"
          aria-label="Query"
        />
        <button type="submit" className="query-bar__send" disabled={loading || !value.trim()}>
          {loading ? (
            <span className="query-bar__send-dots" aria-hidden>
              <span />
              <span />
              <span />
            </span>
          ) : (
            <span>Send</span>
          )}
        </button>
      </form>
    </div>
  );
}
