import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { neighborNodeIds } from "../graphUtils";
import type {
  AskResponse,
  Artifact,
  ArtifactType,
  GraphData,
  GraphNode,
  QuizPublic,
  QuizResult,
  Resource,
  TopicStatus,
  Zone,
} from "../types";

const STATUS_LABEL: Record<TopicStatus, string> = {
  not_started: "Not started",
  in_progress: "In progress",
  complete: "Complete",
};

const STATUS_ORDER: TopicStatus[] = ["not_started", "in_progress", "complete"];
const RESOURCE_TYPES: Resource["type"][] = ["document", "note", "link"];
const ARTIFACT_TYPES: ArtifactType[] = ["note", "code_snippet", "summary", "generated_output"];
const QA_LOG_SEPARATOR = "\n\nA: ";

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
  graphData: GraphData;
  node: GraphNode | null;
  onClose: () => void;
  onNavigateToNode: (node: GraphNode) => void;
  /** Called after a status change, resource attach, quiz pass, or zone change so the caller can refresh graph/stats. */
  onTopicChanged: () => void;
  zones: Zone[];
};

export function NodeDetailsPanel({ graphData, node, onClose, onNavigateToNode, onTopicChanged, zones }: Props) {
  const related = useMemo(() => {
    if (!node) return [];
    const ids = neighborNodeIds(graphData, node.id);
    const byId = new Map(graphData.nodes.map((n) => [n.id, n]));
    return ids.map((id) => byId.get(id)).filter((n): n is GraphNode => Boolean(n));
  }, [graphData, node]);

  const [statusBusy, setStatusBusy] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);

  const [resourceType, setResourceType] = useState<Resource["type"]>("document");
  const [resourceRef, setResourceRef] = useState("");
  const [resourceTitle, setResourceTitle] = useState("");
  const [resourceBusy, setResourceBusy] = useState(false);
  const [resourceError, setResourceError] = useState<string | null>(null);

  const [quiz, setQuiz] = useState<QuizPublic | null>(null);
  const [quizAnswers, setQuizAnswers] = useState<Record<string, number>>({});
  const [quizResult, setQuizResult] = useState<QuizResult | null>(null);
  const [quizBusy, setQuizBusy] = useState(false);
  const [quizError, setQuizError] = useState<string | null>(null);

  const [zoneBusy, setZoneBusy] = useState(false);
  const [zoneError, setZoneError] = useState<string | null>(null);

  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [artifactsLoading, setArtifactsLoading] = useState(false);
  const [artifactType, setArtifactType] = useState<ArtifactType>("note");
  const [artifactTitle, setArtifactTitle] = useState("");
  const [artifactContent, setArtifactContent] = useState("");
  const [artifactBusy, setArtifactBusy] = useState(false);
  const [artifactError, setArtifactError] = useState<string | null>(null);

  const [askQuestion, setAskQuestion] = useState("");
  const [askBusy, setAskBusy] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  useEffect(() => {
    setStatusError(null);
    setResourceRef("");
    setResourceTitle("");
    setResourceError(null);
    setQuiz(null);
    setQuizAnswers({});
    setQuizResult(null);
    setQuizError(null);
    setZoneError(null);
    setArtifacts([]);
    setArtifactTitle("");
    setArtifactContent("");
    setArtifactError(null);
    setAskQuestion("");
    setAskError(null);
  }, [node?.id]);

  const loadArtifacts = useCallback(async (topicId: string) => {
    setArtifactsLoading(true);
    try {
      const list = await fetchJson<Artifact[]>(`/topics/${encodeURIComponent(topicId)}/artifacts`);
      setArtifacts(list);
    } catch {
      setArtifacts([]);
    } finally {
      setArtifactsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (node?.id) void loadArtifacts(node.id);
  }, [node?.id, loadArtifacts]);

  const studyArtifacts = useMemo(() => artifacts.filter((a) => a.type !== "qa_log"), [artifacts]);

  const qaTurns = useMemo(
    () =>
      artifacts
        .filter((a) => a.type === "qa_log")
        .map((a) => {
          const sepIndex = a.content.indexOf(QA_LOG_SEPARATOR);
          return sepIndex === -1
            ? { id: a.id, question: a.title || "(question)", answer: a.content }
            : { id: a.id, question: a.content.slice(3, sepIndex), answer: a.content.slice(sepIndex + QA_LOG_SEPARATOR.length) };
        }),
    [artifacts],
  );

  async function handleSetStatus(next: TopicStatus) {
    if (!node || statusBusy || node.status === next) return;
    setStatusBusy(true);
    setStatusError(null);
    try {
      await fetchJson(`/topics/${encodeURIComponent(node.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: next }),
      });
      onTopicChanged();
    } catch (err) {
      setStatusError(err instanceof Error ? err.message : "Failed to update status");
    } finally {
      setStatusBusy(false);
    }
  }

  async function handleAttachResource(e: FormEvent) {
    e.preventDefault();
    if (!node || resourceBusy || !resourceRef.trim()) return;
    setResourceBusy(true);
    setResourceError(null);
    try {
      await fetchJson(`/topics/${encodeURIComponent(node.id)}/resources`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: resourceType, source_ref: resourceRef.trim(), title: resourceTitle.trim() }),
      });
      setResourceRef("");
      setResourceTitle("");
      onTopicChanged();
    } catch (err) {
      setResourceError(err instanceof Error ? err.message : "Failed to attach resource");
    } finally {
      setResourceBusy(false);
    }
  }

  async function handleSetZone(zoneId: string) {
    if (!node || zoneBusy) return;
    setZoneBusy(true);
    setZoneError(null);
    try {
      await fetchJson(`/topics/${encodeURIComponent(node.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ zone_id: zoneId || null }),
      });
      onTopicChanged();
    } catch (err) {
      setZoneError(err instanceof Error ? err.message : "Failed to update zone");
    } finally {
      setZoneBusy(false);
    }
  }

  async function handleCreateArtifact(e: FormEvent) {
    e.preventDefault();
    if (!node || artifactBusy || !artifactContent.trim()) return;
    setArtifactBusy(true);
    setArtifactError(null);
    try {
      await fetchJson(`/topics/${encodeURIComponent(node.id)}/artifacts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: artifactType, title: artifactTitle.trim(), content: artifactContent.trim() }),
      });
      setArtifactTitle("");
      setArtifactContent("");
      await loadArtifacts(node.id);
    } catch (err) {
      setArtifactError(err instanceof Error ? err.message : "Failed to save artifact");
    } finally {
      setArtifactBusy(false);
    }
  }

  async function handleAsk(e: FormEvent) {
    e.preventDefault();
    if (!node || askBusy || !askQuestion.trim()) return;
    setAskBusy(true);
    setAskError(null);
    try {
      await fetchJson<AskResponse>(`/topics/${encodeURIComponent(node.id)}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: askQuestion.trim() }),
      });
      setAskQuestion("");
      await loadArtifacts(node.id);
    } catch (err) {
      setAskError(err instanceof Error ? err.message : "Failed to get an answer");
    } finally {
      setAskBusy(false);
    }
  }

  async function handleGenerateQuiz() {
    if (!node || quizBusy) return;
    setQuizBusy(true);
    setQuizError(null);
    setQuizResult(null);
    setQuizAnswers({});
    try {
      const q = await fetchJson<QuizPublic>(`/topics/${encodeURIComponent(node.id)}/quiz`, { method: "POST" });
      setQuiz(q);
    } catch (err) {
      setQuizError(err instanceof Error ? err.message : "Failed to generate quiz");
    } finally {
      setQuizBusy(false);
    }
  }

  async function handleSubmitQuiz() {
    if (!node || !quiz || quizBusy) return;
    setQuizBusy(true);
    setQuizError(null);
    try {
      const result = await fetchJson<QuizResult>(`/topics/${encodeURIComponent(node.id)}/quiz/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers: quizAnswers }),
      });
      setQuizResult(result);
      if (result.passed) onTopicChanged();
    } catch (err) {
      setQuizError(err instanceof Error ? err.message : "Failed to submit quiz");
    } finally {
      setQuizBusy(false);
    }
  }

  const allAnswered = Boolean(quiz) && quiz!.questions.every((q) => quizAnswers[q.id] !== undefined);

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
              <span className={`status-pill status-pill--${node.status}`}>{STATUS_LABEL[node.status]}</span>
            ) : null}
            {node.summary ? <p className="node-card__summary">{node.summary}</p> : null}

            <section className="node-card__section">
              <h4>Status</h4>
              <div className="status-toggle">
                {STATUS_ORDER.map((s) => (
                  <button
                    key={s}
                    type="button"
                    className={`status-toggle__btn${node.status === s ? " status-toggle__btn--active" : ""}`}
                    disabled={statusBusy || node.status === s}
                    onClick={() => void handleSetStatus(s)}
                  >
                    {STATUS_LABEL[s]}
                  </button>
                ))}
              </div>
              {statusError ? <p className="modal__error">{statusError}</p> : null}
            </section>

            <section className="node-card__section">
              <h4>Zone</h4>
              <select
                className="resource-form__select"
                style={{ width: "100%" }}
                value={node.zone_id ?? ""}
                onChange={(e) => void handleSetZone(e.target.value)}
                disabled={zoneBusy}
              >
                <option value="">No zone</option>
                {zones.map((z) => (
                  <option key={z.id} value={z.id}>
                    {z.label}
                  </option>
                ))}
              </select>
              {zoneError ? <p className="modal__error">{zoneError}</p> : null}
            </section>

            <section className="node-card__section">
              <h4>Resources</h4>
              {node.resources && node.resources.length > 0 ? (
                <ul className="node-card__sources">
                  {node.resources.map((r) => (
                    <li key={r.id}>
                      [{r.type}] {r.title || r.source_ref}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="sidebar__muted">No resources attached yet.</p>
              )}
              <form className="resource-form" onSubmit={handleAttachResource}>
                <select
                  className="resource-form__select"
                  value={resourceType}
                  onChange={(e) => setResourceType(e.target.value as Resource["type"])}
                  disabled={resourceBusy}
                >
                  {RESOURCE_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
                <input
                  className="resource-form__input"
                  placeholder={resourceType === "link" ? "URL" : "ingested note filename (see /ingest)"}
                  value={resourceRef}
                  onChange={(e) => setResourceRef(e.target.value)}
                  disabled={resourceBusy}
                />
                <input
                  className="resource-form__input"
                  placeholder="Title (optional)"
                  value={resourceTitle}
                  onChange={(e) => setResourceTitle(e.target.value)}
                  disabled={resourceBusy}
                />
                <button type="submit" className="resource-form__btn" disabled={resourceBusy || !resourceRef.trim()}>
                  {resourceBusy ? "Attaching…" : "Attach"}
                </button>
              </form>
              {resourceError ? <p className="modal__error">{resourceError}</p> : null}
            </section>

            <section className="node-card__section">
              <h4>Artifacts</h4>
              {artifactsLoading ? (
                <p className="sidebar__muted">Loading…</p>
              ) : studyArtifacts.length > 0 ? (
                <ul className="node-card__sources">
                  {studyArtifacts.map((a) => (
                    <li key={a.id}>
                      [{a.type}] {a.title || "(untitled)"}: {a.content.length > 80 ? `${a.content.slice(0, 80)}…` : a.content}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="sidebar__muted">Nothing produced for this topic yet.</p>
              )}
              <form onSubmit={handleCreateArtifact}>
                <div className="resource-form">
                  <select
                    className="resource-form__select"
                    value={artifactType}
                    onChange={(e) => setArtifactType(e.target.value as ArtifactType)}
                    disabled={artifactBusy}
                  >
                    {ARTIFACT_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t.replace(/_/g, " ")}
                      </option>
                    ))}
                  </select>
                  <input
                    className="resource-form__input"
                    placeholder="Title (optional)"
                    value={artifactTitle}
                    onChange={(e) => setArtifactTitle(e.target.value)}
                    disabled={artifactBusy}
                  />
                </div>
                <textarea
                  className="modal__textarea"
                  placeholder="What you produced -- a note, snippet, summary…"
                  value={artifactContent}
                  onChange={(e) => setArtifactContent(e.target.value)}
                  rows={2}
                  disabled={artifactBusy}
                  style={{ marginTop: "0.4rem" }}
                />
                <button
                  type="submit"
                  className="resource-form__btn"
                  style={{ marginTop: "0.4rem" }}
                  disabled={artifactBusy || !artifactContent.trim()}
                >
                  {artifactBusy ? "Saving…" : "Save artifact"}
                </button>
              </form>
              {artifactError ? <p className="modal__error">{artifactError}</p> : null}
            </section>

            <section className="node-card__section">
              <h4>Ask about this topic</h4>
              {qaTurns.length > 0 ? (
                <ul className="qa-log">
                  {qaTurns.map((t) => (
                    <li key={t.id} className="qa-log__turn">
                      <p className="qa-log__question">{t.question}</p>
                      <p className="qa-log__answer">{t.answer}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="sidebar__muted">No questions asked yet -- ask anything about this topic.</p>
              )}
              <form className="qa-log__form" onSubmit={handleAsk}>
                <textarea
                  className="modal__textarea"
                  placeholder="Ask a question about this topic…"
                  value={askQuestion}
                  onChange={(e) => setAskQuestion(e.target.value)}
                  rows={2}
                  disabled={askBusy}
                />
                <button
                  type="submit"
                  className="resource-form__btn"
                  style={{ marginTop: "0.4rem" }}
                  disabled={askBusy || !askQuestion.trim()}
                >
                  {askBusy ? "Thinking…" : "Ask"}
                </button>
              </form>
              {askError ? <p className="modal__error">{askError}</p> : null}
            </section>

            <section className="node-card__section">
              <h4>Closure quiz</h4>
              {!quiz ? (
                <button type="button" className="sidebar__add-note" onClick={() => void handleGenerateQuiz()} disabled={quizBusy}>
                  {quizBusy ? "Generating…" : "Generate quiz"}
                </button>
              ) : (
                <div className="quiz">
                  {quiz.questions.map((q, qi) => (
                    <fieldset className="quiz__question" key={q.id}>
                      <legend>
                        {qi + 1}. {q.question}
                      </legend>
                      {q.choices.map((choice, ci) => {
                        const resultForQ = quizResult?.results.find((r) => r.question_id === q.id);
                        const showResult = Boolean(resultForQ);
                        const isCorrectChoice = resultForQ?.correct_index === ci;
                        const isSelected = quizAnswers[q.id] === ci;
                        return (
                          <label
                            key={ci}
                            className={`quiz__choice${showResult && isCorrectChoice ? " quiz__choice--correct" : ""}${
                              showResult && isSelected && !isCorrectChoice ? " quiz__choice--wrong" : ""
                            }`}
                          >
                            <input
                              type="radio"
                              name={q.id}
                              checked={isSelected}
                              disabled={Boolean(quizResult)}
                              onChange={() => setQuizAnswers((a) => ({ ...a, [q.id]: ci }))}
                            />
                            {choice}
                          </label>
                        );
                      })}
                    </fieldset>
                  ))}

                  {quizResult ? (
                    <div className={`quiz__result${quizResult.passed ? " quiz__result--pass" : " quiz__result--fail"}`}>
                      {quizResult.correct_count}/{quizResult.total} correct ({Math.round(quizResult.score * 100)}%) --{" "}
                      {quizResult.passed ? "passed" : "not passed yet"}
                    </div>
                  ) : null}

                  <div className="quiz__actions">
                    {!quizResult ? (
                      <button
                        type="button"
                        className="modal__btn modal__btn--primary"
                        onClick={() => void handleSubmitQuiz()}
                        disabled={quizBusy || !allAnswered}
                      >
                        {quizBusy ? "Submitting…" : "Submit answers"}
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="modal__btn modal__btn--ghost"
                        onClick={() => void handleGenerateQuiz()}
                        disabled={quizBusy}
                      >
                        Generate new quiz
                      </button>
                    )}
                  </div>
                </div>
              )}
              {quizError ? <p className="modal__error">{quizError}</p> : null}
            </section>

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
