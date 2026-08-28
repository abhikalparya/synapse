import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type {
  Artifact,
  ArtifactType,
  AskResponse,
  Dependency,
  GraphNode,
  PathResponse,
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
  topic: GraphNode;
  nodes: GraphNode[];
  dependencies: Dependency[];
  dependenciesLoading: boolean;
  dependenciesError: string | null;
  zones: Zone[];
  onBack: () => void;
  onOpenTopic: (id: string) => void;
  onOpenExplore: (id: string) => void;
  onTopicChanged: () => void;
};

export function TopicLearningWorkspace({
  topic,
  nodes,
  dependencies,
  dependenciesLoading,
  dependenciesError,
  zones,
  onBack,
  onOpenTopic,
  onOpenExplore,
  onTopicChanged,
}: Props) {
  const [path, setPath] = useState<PathResponse | null>(null);
  const [pathLoading, setPathLoading] = useState(true);
  const [pathError, setPathError] = useState<string | null>(null);

  const [statusBusy, setStatusBusy] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);

  const [resourceType, setResourceType] = useState<Resource["type"]>("document");
  const [resourceRef, setResourceRef] = useState("");
  const [resourceTitle, setResourceTitle] = useState("");
  const [resourceBusy, setResourceBusy] = useState(false);
  const [resourceError, setResourceError] = useState<string | null>(null);

  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [artifactsLoading, setArtifactsLoading] = useState(true);
  const [artifactType, setArtifactType] = useState<ArtifactType>("note");
  const [artifactTitle, setArtifactTitle] = useState("");
  const [artifactContent, setArtifactContent] = useState("");
  const [artifactBusy, setArtifactBusy] = useState(false);
  const [artifactError, setArtifactError] = useState<string | null>(null);

  const [askQuestion, setAskQuestion] = useState("");
  const [askBusy, setAskBusy] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  const [quiz, setQuiz] = useState<QuizPublic | null>(null);
  const [quizAnswers, setQuizAnswers] = useState<Record<string, number>>({});
  const [quizResult, setQuizResult] = useState<QuizResult | null>(null);
  const [quizBusy, setQuizBusy] = useState(false);
  const [quizError, setQuizError] = useState<string | null>(null);
  const mountedRef = useRef(false);
  const activeTopicIdRef = useRef(topic.id);
  const artifactsRequestRef = useRef(0);
  activeTopicIdRef.current = topic.id;

  const nodeById = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);
  const zone = topic.zone_id ? zones.find((candidate) => candidate.id === topic.zone_id) : undefined;
  const isActiveTopic = useCallback(
    (topicId: string) => mountedRef.current && activeTopicIdRef.current === topicId,
    [],
  );

  const loadArtifacts = useCallback(async (topicId: string) => {
    if (!isActiveTopic(topicId)) return;
    const requestId = artifactsRequestRef.current + 1;
    artifactsRequestRef.current = requestId;
    setArtifactsLoading(true);
    try {
      const list = await fetchJson<Artifact[]>(`/topics/${encodeURIComponent(topicId)}/artifacts`);
      if (requestId === artifactsRequestRef.current && isActiveTopic(topicId)) setArtifacts(list);
    } catch {
      if (requestId === artifactsRequestRef.current && isActiveTopic(topicId)) setArtifacts([]);
    } finally {
      if (requestId === artifactsRequestRef.current && isActiveTopic(topicId)) setArtifactsLoading(false);
    }
  }, [isActiveTopic]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setPath(null);
    setPathError(null);
    setPathLoading(true);
    void fetchJson<PathResponse>(`/graph/path?target=${encodeURIComponent(topic.id)}`)
      .then((result) => {
        if (!cancelled) setPath(result);
      })
      .catch((err) => {
        if (!cancelled) setPathError(err instanceof Error ? err.message : "Failed to load prerequisites");
      })
      .finally(() => {
        if (!cancelled) setPathLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [topic.id]);

  useEffect(() => {
    setStatusError(null);
    setStatusBusy(false);
    setResourceRef("");
    setResourceTitle("");
    setResourceError(null);
    setResourceBusy(false);
    setArtifacts([]);
    setArtifactsLoading(true);
    setArtifactTitle("");
    setArtifactContent("");
    setArtifactError(null);
    setArtifactBusy(false);
    setAskQuestion("");
    setAskError(null);
    setAskBusy(false);
    setQuiz(null);
    setQuizAnswers({});
    setQuizResult(null);
    setQuizError(null);
    setQuizBusy(false);
    void loadArtifacts(topic.id);
  }, [loadArtifacts, topic.id]);

  const studyArtifacts = useMemo(() => artifacts.filter((artifact) => artifact.type !== "qa_log"), [artifacts]);
  const qaTurns = useMemo(
    () =>
      artifacts
        .filter((artifact) => artifact.type === "qa_log")
        .map((artifact) => {
          const separatorIndex = artifact.content.indexOf(QA_LOG_SEPARATOR);
          return separatorIndex === -1
            ? { id: artifact.id, question: artifact.title || "(question)", answer: artifact.content }
            : {
                id: artifact.id,
                question: artifact.content.slice(3, separatorIndex),
                answer: artifact.content.slice(separatorIndex + QA_LOG_SEPARATOR.length),
              };
        }),
    [artifacts],
  );

  const directPrerequisites = useMemo(
    () =>
      dependencies
        .filter((dependency) => dependency.from_topic_id === topic.id)
        .map((dependency) => nodeById.get(dependency.to_topic_id))
        .filter((node): node is GraphNode => Boolean(node)),
    [dependencies, nodeById, topic.id],
  );

  const dependents = useMemo(
    () =>
      dependencies
        .filter((dependency) => dependency.to_topic_id === topic.id)
        .map((dependency) => nodeById.get(dependency.from_topic_id))
        .filter((node): node is GraphNode => Boolean(node)),
    [dependencies, nodeById, topic.id],
  );

  const orderedPrerequisites = useMemo(() => {
    if (!path) return directPrerequisites;
    return path.chain
      .filter((entry) => entry.id !== topic.id)
      .map((entry) => nodeById.get(entry.id))
      .filter((node): node is GraphNode => Boolean(node));
  }, [directPrerequisites, nodeById, path, topic.id]);

  async function handleSetStatus(next: TopicStatus) {
    if (statusBusy || topic.status === next) return;
    const requestTopicId = topic.id;
    setStatusBusy(true);
    setStatusError(null);
    try {
      await fetchJson(`/topics/${encodeURIComponent(requestTopicId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: next }),
      });
      if (isActiveTopic(requestTopicId)) onTopicChanged();
    } catch (err) {
      if (isActiveTopic(requestTopicId)) {
        setStatusError(err instanceof Error ? err.message : "Failed to update status");
      }
    } finally {
      if (isActiveTopic(requestTopicId)) setStatusBusy(false);
    }
  }

  async function handleAttachResource(event: FormEvent) {
    event.preventDefault();
    if (resourceBusy || !resourceRef.trim()) return;
    const requestTopicId = topic.id;
    setResourceBusy(true);
    setResourceError(null);
    try {
      await fetchJson(`/topics/${encodeURIComponent(requestTopicId)}/resources`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: resourceType,
          source_ref: resourceRef.trim(),
          title: resourceTitle.trim(),
        }),
      });
      if (!isActiveTopic(requestTopicId)) return;
      setResourceRef("");
      setResourceTitle("");
      onTopicChanged();
    } catch (err) {
      if (isActiveTopic(requestTopicId)) {
        setResourceError(err instanceof Error ? err.message : "Failed to attach resource");
      }
    } finally {
      if (isActiveTopic(requestTopicId)) setResourceBusy(false);
    }
  }

  async function handleCreateArtifact(event: FormEvent) {
    event.preventDefault();
    if (artifactBusy || !artifactContent.trim()) return;
    const requestTopicId = topic.id;
    setArtifactBusy(true);
    setArtifactError(null);
    try {
      await fetchJson(`/topics/${encodeURIComponent(requestTopicId)}/artifacts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: artifactType,
          title: artifactTitle.trim(),
          content: artifactContent.trim(),
        }),
      });
      if (!isActiveTopic(requestTopicId)) return;
      setArtifactTitle("");
      setArtifactContent("");
      await loadArtifacts(requestTopicId);
    } catch (err) {
      if (isActiveTopic(requestTopicId)) {
        setArtifactError(err instanceof Error ? err.message : "Failed to save artifact");
      }
    } finally {
      if (isActiveTopic(requestTopicId)) setArtifactBusy(false);
    }
  }

  async function handleAsk(event: FormEvent) {
    event.preventDefault();
    if (askBusy || !askQuestion.trim()) return;
    const requestTopicId = topic.id;
    setAskBusy(true);
    setAskError(null);
    try {
      await fetchJson<AskResponse>(`/topics/${encodeURIComponent(requestTopicId)}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: askQuestion.trim() }),
      });
      if (!isActiveTopic(requestTopicId)) return;
      setAskQuestion("");
      await loadArtifacts(requestTopicId);
    } catch (err) {
      if (isActiveTopic(requestTopicId)) {
        setAskError(err instanceof Error ? err.message : "Failed to get an answer");
      }
    } finally {
      if (isActiveTopic(requestTopicId)) setAskBusy(false);
    }
  }

  async function handleGenerateQuiz() {
    if (quizBusy) return;
    const requestTopicId = topic.id;
    setQuizBusy(true);
    setQuizError(null);
    setQuizResult(null);
    setQuizAnswers({});
    try {
      const generated = await fetchJson<QuizPublic>(`/topics/${encodeURIComponent(requestTopicId)}/quiz`, { method: "POST" });
      if (isActiveTopic(requestTopicId)) setQuiz(generated);
    } catch (err) {
      if (isActiveTopic(requestTopicId)) {
        setQuizError(err instanceof Error ? err.message : "Failed to generate quiz");
      }
    } finally {
      if (isActiveTopic(requestTopicId)) setQuizBusy(false);
    }
  }

  async function handleSubmitQuiz() {
    if (!quiz || quizBusy) return;
    const requestTopicId = topic.id;
    setQuizBusy(true);
    setQuizError(null);
    try {
      const result = await fetchJson<QuizResult>(`/topics/${encodeURIComponent(requestTopicId)}/quiz/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answers: quizAnswers }),
      });
      if (!isActiveTopic(requestTopicId)) return;
      setQuizResult(result);
      if (result.passed) onTopicChanged();
    } catch (err) {
      if (isActiveTopic(requestTopicId)) {
        setQuizError(err instanceof Error ? err.message : "Failed to submit quiz");
      }
    } finally {
      if (isActiveTopic(requestTopicId)) setQuizBusy(false);
    }
  }

  const allAnswered = Boolean(quiz) && quiz!.questions.every((question) => quizAnswers[question.id] !== undefined);

  return (
    <article className="topic-learning" aria-labelledby="topic-learning-title">
      <div className="topic-learning__document">
        <header className="topic-learning__header">
          <div>
            <p className="workspace-view__eyebrow">Learn / Topic</p>
            <h1 id="topic-learning-title">{topic.title ?? topic.id}</h1>
            {topic.summary ? <p className="topic-learning__summary">{topic.summary}</p> : null}
            <div className="topic-learning__meta">
              <span className={`status-pill status-pill--${topic.status ?? "not_started"}`}>
                {STATUS_LABEL[topic.status ?? "not_started"]}
              </span>
              {zone ? <span className="topic-learning__zone">{zone.label}</span> : null}
            </div>
            <div className="topic-learning__status-control">
              <span className="topic-learning__status-label">Update status</span>
              <div className="status-toggle">
                {STATUS_ORDER.map((status) => (
                  <button
                    key={status}
                    type="button"
                    className={`status-toggle__btn${topic.status === status ? " status-toggle__btn--active" : ""}`}
                    disabled={statusBusy || topic.status === status}
                    onClick={() => void handleSetStatus(status)}
                  >
                    {STATUS_LABEL[status]}
                  </button>
                ))}
              </div>
              {statusError ? <p className="modal__error">{statusError}</p> : null}
            </div>
          </div>
          <div className="topic-learning__actions">
            <button type="button" className="modal__btn modal__btn--ghost" onClick={onBack}>
              All topics
            </button>
            <button type="button" className="modal__btn modal__btn--primary" onClick={() => onOpenExplore(topic.id)}>
              Open in Explore
            </button>
          </div>
        </header>

        <section className="topic-learning__section" aria-labelledby="topic-context-title">
          <div className="topic-learning__section-heading">
            <div>
              <p className="topic-learning__kicker">Learning context</p>
              <h2 id="topic-context-title">Where this topic fits</h2>
            </div>
            <span className="topic-learning__section-note">Directed prerequisites</span>
          </div>
          {dependenciesLoading || pathLoading ? <p className="topic-learning__muted">Loading relationships…</p> : null}
          {dependenciesError ? <p className="modal__error">{dependenciesError}</p> : null}
          {pathError ? <p className="topic-learning__muted">The ordered prerequisite path is unavailable.</p> : null}
          {!dependenciesLoading && !pathLoading ? (
            <div className="topic-learning__relationship-grid">
              <div>
                <h3>Prerequisite path</h3>
                {orderedPrerequisites.length ? (
                  <ol className="topic-learning__topic-list">
                    {orderedPrerequisites.map((node) => (
                      <li key={node.id}>
                        <button type="button" onClick={() => onOpenTopic(node.id)}>
                          <span>{node.title ?? node.id}</span>
                          <small>{STATUS_LABEL[node.status ?? "not_started"]}</small>
                        </button>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="topic-learning__muted">This topic has no recorded prerequisites.</p>
                )}
              </div>
              <div>
                <h3>Topics that depend on this</h3>
                {dependenciesError ? (
                  <p className="topic-learning__muted">Dependent topics are unavailable.</p>
                ) : dependents.length ? (
                  <ul className="topic-learning__topic-list">
                    {dependents.map((node) => (
                      <li key={node.id}>
                        <button type="button" onClick={() => onOpenTopic(node.id)}>
                          <span>{node.title ?? node.id}</span>
                          <small>{STATUS_LABEL[node.status ?? "not_started"]}</small>
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="topic-learning__muted">No recorded dependent topics.</p>
                )}
              </div>
            </div>
          ) : null}
        </section>

        <section className="topic-learning__section" aria-labelledby="topic-material-title">
          <div className="topic-learning__section-heading">
            <div>
              <p className="topic-learning__kicker">Learning material</p>
              <h2 id="topic-material-title">Resources and notes</h2>
            </div>
          </div>
          {topic.resources?.length ? (
            <ul className="topic-learning__resource-list">
              {topic.resources.map((resource) => (
                <li key={resource.id}>
                  <span className="topic-learning__resource-type">{resource.type}</span>
                  {resource.type === "link" ? (
                    <a href={resource.source_ref} target="_blank" rel="noreferrer">
                      {resource.title || resource.source_ref}
                    </a>
                  ) : (
                    <span>{resource.title || resource.source_ref}</span>
                  )}
                </li>
              ))}
            </ul>
          ) : (
            <p className="topic-learning__muted">No resources attached yet.</p>
          )}
          <form className="topic-learning__inline-form" onSubmit={handleAttachResource}>
            <select
              className="resource-form__select"
              value={resourceType}
              onChange={(event) => setResourceType(event.target.value as Resource["type"])}
              disabled={resourceBusy}
              aria-label="Resource type"
            >
              {RESOURCE_TYPES.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
            <input
              className="resource-form__input"
              placeholder={resourceType === "link" ? "URL" : "ingested note filename"}
              value={resourceRef}
              onChange={(event) => setResourceRef(event.target.value)}
              disabled={resourceBusy}
            />
            <input
              className="resource-form__input"
              placeholder="Title (optional)"
              value={resourceTitle}
              onChange={(event) => setResourceTitle(event.target.value)}
              disabled={resourceBusy}
            />
            <button type="submit" className="resource-form__btn" disabled={resourceBusy || !resourceRef.trim()}>
              {resourceBusy ? "Attaching…" : "Attach"}
            </button>
          </form>
          {resourceError ? <p className="modal__error">{resourceError}</p> : null}

          <div className="topic-learning__subsection">
            <h3>Artifacts</h3>
            {artifactsLoading ? (
              <p className="topic-learning__muted">Loading artifacts…</p>
            ) : studyArtifacts.length ? (
              <ul className="topic-learning__artifact-list">
                {studyArtifacts.map((artifact) => (
                  <li key={artifact.id}>
                    <div className="topic-learning__artifact-heading">
                      <span>{artifact.title || "(untitled)"}</span>
                      <small>{artifact.type.replace(/_/g, " ")}</small>
                    </div>
                    <p>{artifact.content}</p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="topic-learning__muted">No study notes or artifacts yet.</p>
            )}
            <form className="topic-learning__artifact-form" onSubmit={handleCreateArtifact}>
              <div className="topic-learning__artifact-fields">
                <select
                  className="resource-form__select"
                  value={artifactType}
                  onChange={(event) => setArtifactType(event.target.value as ArtifactType)}
                  disabled={artifactBusy}
                  aria-label="Artifact type"
                >
                  {ARTIFACT_TYPES.map((type) => (
                    <option key={type} value={type}>
                      {type.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
                <input
                  className="resource-form__input"
                  placeholder="Title (optional)"
                  value={artifactTitle}
                  onChange={(event) => setArtifactTitle(event.target.value)}
                  disabled={artifactBusy}
                />
              </div>
              <textarea
                className="modal__textarea"
                placeholder="Write a note, summary, snippet, or other study output…"
                value={artifactContent}
                onChange={(event) => setArtifactContent(event.target.value)}
                rows={3}
                disabled={artifactBusy}
              />
              <button type="submit" className="resource-form__btn" disabled={artifactBusy || !artifactContent.trim()}>
                {artifactBusy ? "Saving…" : "Save artifact"}
              </button>
            </form>
            {artifactError ? <p className="modal__error">{artifactError}</p> : null}
          </div>
        </section>

        <section className="topic-learning__section" aria-labelledby="topic-ask-title">
          <div className="topic-learning__section-heading">
            <div>
              <p className="topic-learning__kicker">Understand this topic</p>
              <h2 id="topic-ask-title">Ask a focused question</h2>
            </div>
          </div>
          {qaTurns.length ? (
            <ul className="qa-log topic-learning__qa-log">
              {qaTurns.map((turn) => (
                <li key={turn.id} className="qa-log__turn">
                  <p className="qa-log__question">{turn.question}</p>
                  <p className="qa-log__answer">{turn.answer}</p>
                </li>
              ))}
            </ul>
          ) : (
            <p className="topic-learning__muted">No questions asked yet.</p>
          )}
          <form className="qa-log__form" onSubmit={handleAsk}>
            <textarea
              className="modal__textarea"
              placeholder="Ask a question about this topic…"
              value={askQuestion}
              onChange={(event) => setAskQuestion(event.target.value)}
              rows={3}
              disabled={askBusy}
            />
            <button type="submit" className="resource-form__btn" disabled={askBusy || !askQuestion.trim()}>
              {askBusy ? "Thinking…" : "Ask"}
            </button>
          </form>
          {askError ? <p className="modal__error">{askError}</p> : null}
        </section>

        <section className="topic-learning__section" aria-labelledby="topic-quiz-title">
          <div className="topic-learning__section-heading">
            <div>
              <p className="topic-learning__kicker">Check understanding</p>
              <h2 id="topic-quiz-title">Closure quiz</h2>
            </div>
            {topic.quiz_passed ? <span className="topic-learning__section-note">Passed</span> : null}
          </div>
          {!quiz ? (
            <>
              <p className="topic-learning__muted">Generate a short quiz from this topic’s summary and readable resources.</p>
              <button type="button" className="resource-form__btn" onClick={() => void handleGenerateQuiz()} disabled={quizBusy}>
                {quizBusy ? "Generating…" : "Generate quiz"}
              </button>
            </>
          ) : (
            <div className="quiz">
              {quiz.questions.map((question, questionIndex) => (
                <fieldset className="quiz__question" key={question.id}>
                  <legend>
                    {questionIndex + 1}. {question.question}
                  </legend>
                  {question.choices.map((choice, choiceIndex) => {
                    const resultForQuestion = quizResult?.results.find((result) => result.question_id === question.id);
                    const showResult = Boolean(resultForQuestion);
                    const isCorrectChoice = resultForQuestion?.correct_index === choiceIndex;
                    const isSelected = quizAnswers[question.id] === choiceIndex;
                    return (
                      <label
                        key={choiceIndex}
                        className={`quiz__choice${showResult && isCorrectChoice ? " quiz__choice--correct" : ""}${
                          showResult && isSelected && !isCorrectChoice ? " quiz__choice--wrong" : ""
                        }`}
                      >
                        <input
                          type="radio"
                          name={question.id}
                          checked={isSelected}
                          disabled={Boolean(quizResult)}
                          onChange={() => setQuizAnswers((answers) => ({ ...answers, [question.id]: choiceIndex }))}
                        />
                        {choice}
                      </label>
                    );
                  })}
                </fieldset>
              ))}
              {quizResult ? (
                <div className={`quiz__result${quizResult.passed ? " quiz__result--pass" : " quiz__result--fail"}`}>
                  {quizResult.correct_count}/{quizResult.total} correct ({Math.round(quizResult.score * 100)}%) —{" "}
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
                  <button type="button" className="modal__btn modal__btn--ghost" onClick={() => void handleGenerateQuiz()} disabled={quizBusy}>
                    Generate new quiz
                  </button>
                )}
              </div>
            </div>
          )}
          {quizError ? <p className="modal__error">{quizError}</p> : null}
        </section>
      </div>
    </article>
  );
}
