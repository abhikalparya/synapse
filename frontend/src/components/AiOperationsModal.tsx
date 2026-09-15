import { useCallback, useId, useRef, useState, type FormEvent } from "react";
import type { ApplyResponse, AuditReport, GraphNode, Proposal } from "../types";
import { FileTypeIcon, fileExt } from "./FileTypeIcon";
import { ProposalDetails } from "./ProposalDetails";

type Mode = "ingest" | "expand" | "audit" | "reshape" | "obsidian";

const MODE_LABEL: Record<Mode, string> = {
  ingest: "Ingest",
  expand: "Expand",
  audit: "Audit",
  reshape: "Reshape",
  obsidian: "Obsidian",
};

type Props = {
  open: boolean;
  onClose: () => void;
  nodes: GraphNode[];
  /** Called after a successful apply so the caller can refresh persisted workspace state. */
  onApplied: (result: ApplyResponse) => Promise<void> | void;
  /** Called after a successful discard so the caller can refresh persisted workspace state. */
  onDiscarded: () => Promise<void> | void;
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

const INGEST_ACCEPT =
  ".txt,.md,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";
const INGEST_ACCEPT_EXT = new Set([".txt", ".md", ".pdf", ".docx"]);

function fileKey(f: File): string {
  return `${f.name}:${f.size}:${f.lastModified}`;
}

type IngestUploadJson = {
  status: "ok" | "warning";
  filename?: string | null;
};

type BatchIngestItem = {
  filename: string;
  status: "ok" | "warning" | "error";
  saved_filename?: string | null;
  detail?: string | null;
};

type BatchIngestResponse = {
  items: BatchIngestItem[];
};

export function AiOperationsModal({ open, onClose, nodes, onApplied, onDiscarded }: Props) {
  const idBase = useId();
  const [mode, setMode] = useState<Mode>("ingest");

  const [goal, setGoal] = useState("");
  const [ingestFiles, setIngestFiles] = useState<File[]>([]);
  const [ingestDragOver, setIngestDragOver] = useState(false);
  const ingestFileInputRef = useRef<HTMLInputElement>(null);
  const [expandTopicId, setExpandTopicId] = useState("");
  const [expandInstructions, setExpandInstructions] = useState("");
  const [reshapeTopicIds, setReshapeTopicIds] = useState<Set<string>>(new Set());
  const [reshapeInstructions, setReshapeInstructions] = useState("");
  const [vaultPath, setVaultPath] = useState("");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [auditReport, setAuditReport] = useState<AuditReport | null>(null);

  const resetAll = useCallback(() => {
    setMode("ingest");
    setGoal("");
    setIngestFiles([]);
    setIngestDragOver(false);
    if (ingestFileInputRef.current) ingestFileInputRef.current.value = "";
    setExpandTopicId("");
    setExpandInstructions("");
    setReshapeTopicIds(new Set());
    setReshapeInstructions("");
    setVaultPath("");
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
      setIngestFiles([]);
      setIngestDragOver(false);
      if (ingestFileInputRef.current) ingestFileInputRef.current.value = "";
    },
    [busy],
  );

  const mergeValidatedIngestFiles = useCallback((existing: File[], incoming: File[]) => {
    const rejected: string[] = [];
    const seen = new Set(existing.map(fileKey));
    const next = [...existing];
    for (const f of incoming) {
      const ext = fileExt(f.name);
      if (!INGEST_ACCEPT_EXT.has(ext)) {
        rejected.push(f.name);
        continue;
      }
      const k = fileKey(f);
      if (seen.has(k)) continue;
      seen.add(k);
      next.push(f);
    }
    return { next, rejected };
  }, []);

  const addIngestFiles = useCallback(
    (list: FileList | File[] | null) => {
      if (!list?.length) return;
      const incoming = Array.from(list as FileList);
      setIngestFiles((prev) => {
        const { next, rejected } = mergeValidatedIngestFiles(prev, incoming);
        if (rejected.length) {
          setError(`Skipped unsupported type: ${rejected.slice(0, 4).join(", ")}${rejected.length > 4 ? "…" : ""} (.txt, .md, .pdf, .docx only)`);
        } else {
          setError(null);
        }
        return next;
      });
      if (ingestFileInputRef.current) ingestFileInputRef.current.value = "";
    },
    [mergeValidatedIngestFiles],
  );

  const removeIngestFileAt = useCallback((index: number) => {
    setIngestFiles((prev) => prev.filter((_, i) => i !== index));
    setError(null);
  }, []);

  async function uploadIngestFiles(files: File[]): Promise<string[]> {
    if (files.length === 1) {
      const form = new FormData();
      form.append("file", files[0], files[0].name);
      const up = await fetchJson<IngestUploadJson>("/ingest/upload", { method: "POST", body: form });
      return up.filename ? [up.filename] : [];
    }
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    const batch = await fetchJson<BatchIngestResponse>("/ingest/upload/batch", { method: "POST", body: form });
    const ok = batch.items.filter((i) => i.status !== "error");
    const failed = batch.items.filter((i) => i.status === "error");
    if (ok.length === 0) {
      throw new Error(failed.map((i) => `${i.filename}: ${i.detail ?? "failed"}`).join("; ") || "No files could be saved");
    }
    return ok.map((i) => i.saved_filename).filter((n): n is string => Boolean(n));
  }

  async function handleIngest(e: FormEvent) {
    e.preventDefault();
    const trimmed = goal.trim();
    if ((!trimmed && ingestFiles.length === 0) || busy) return;
    setBusy(true);
    setError(null);
    try {
      const filenames = ingestFiles.length ? await uploadIngestFiles(ingestFiles) : [];
      const result = await fetchJson<Proposal>("/ai/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: trimmed || null, filenames: filenames.length ? filenames : null }),
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

  async function handleObsidianImport(e: FormEvent) {
    e.preventDefault();
    const trimmed = vaultPath.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await fetchJson<Proposal>("/obsidian/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vault_path: trimmed }),
      });
      setProposal(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to import vault");
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
      await onApplied(result);
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
      await onDiscarded();
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
    ...(proposal?.topics.map((topic) => [topic.temp_id, topic.title] as const) ?? []),
    ...nodes.map((node) => [node.id, node.title ?? node.id] as const),
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
              <label className="modal__label" id={`${idBase}-ingest-upload-label`}>
                Files (optional)
              </label>
              <input
                ref={ingestFileInputRef}
                id={`${idBase}-ingest-file`}
                type="file"
                className="visually-hidden"
                accept={INGEST_ACCEPT}
                multiple
                disabled={busy}
                tabIndex={-1}
                aria-labelledby={`${idBase}-ingest-upload-label`}
                onChange={(e) => addIngestFiles(e.target.files)}
              />
              <div
                className={`ingest-drop ${ingestDragOver ? "ingest-drop--active" : ""} ${ingestFiles.length ? "ingest-drop--has-file" : ""}`}
                role="group"
                aria-label="Drop documents here or browse to add files"
                onDragEnter={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIngestDragOver(true);
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIngestDragOver(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIngestDragOver(false);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIngestDragOver(false);
                  addIngestFiles(e.dataTransfer.files);
                }}
              >
                {ingestFiles.length === 0 ? (
                  <div
                    className="ingest-drop__empty"
                    role="button"
                    tabIndex={0}
                    onClick={() => !busy && ingestFileInputRef.current?.click()}
                    onKeyDown={(e) => {
                      if (e.key !== "Enter" && e.key !== " ") return;
                      e.preventDefault();
                      if (!busy) ingestFileInputRef.current?.click();
                    }}
                  >
                    <FileTypeIcon ext="" size="lg" placeholder />
                    <p className="ingest-drop__title">Drop files here</p>
                    <p className="ingest-drop__sub">.txt · .md · .pdf · .docx — multiple files OK</p>
                    <button
                      type="button"
                      className="ingest-drop__browse"
                      onClick={(e) => {
                        e.stopPropagation();
                        ingestFileInputRef.current?.click();
                      }}
                    >
                      Browse files
                    </button>
                  </div>
                ) : (
                  <div className="ingest-drop__multi" onClick={(e) => e.stopPropagation()}>
                    <ul className="ingest-file-list" aria-label="Selected files">
                      {ingestFiles.map((f, i) => (
                        <li key={fileKey(f)} className="ingest-file-list__row">
                          <FileTypeIcon ext={fileExt(f.name)} size="md" />
                          <div className="ingest-file-list__meta">
                            <span className="ingest-file-list__name" title={f.name}>
                              {f.name}
                            </span>
                            <span className="ingest-file-list__sub">{(f.size / 1024).toFixed(1)} KB</span>
                          </div>
                          <button
                            type="button"
                            className="ingest-drop__clear"
                            onClick={() => removeIngestFileAt(i)}
                            disabled={busy}
                            aria-label={`Remove ${f.name}`}
                          >
                            ×
                          </button>
                        </li>
                      ))}
                    </ul>
                    <div className="ingest-drop__multi-actions">
                      <button
                        type="button"
                        className="ingest-drop__change"
                        onClick={() => ingestFileInputRef.current?.click()}
                        disabled={busy}
                      >
                        Add more files
                      </button>
                      <button
                        type="button"
                        className="ingest-drop__clear-all"
                        onClick={() => {
                          setIngestFiles([]);
                          if (ingestFileInputRef.current) ingestFileInputRef.current.value = "";
                        }}
                        disabled={busy}
                      >
                        Clear all
                      </button>
                    </div>
                  </div>
                )}
              </div>

              <label className="modal__label" htmlFor={`${idBase}-goal`}>
                Learning goal{ingestFiles.length ? " (optional)" : ""}
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
                <button
                  type="submit"
                  className="modal__btn modal__btn--primary"
                  disabled={busy || (!goal.trim() && ingestFiles.length === 0)}
                >
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

          {mode === "obsidian" && !proposal ? (
            <form onSubmit={handleObsidianImport}>
              <label className="modal__label" htmlFor={`${idBase}-vault-path`}>
                Vault path
              </label>
              <input
                id={`${idBase}-vault-path`}
                className="resource-form__input"
                style={{ width: "100%" }}
                value={vaultPath}
                onChange={(e) => setVaultPath(e.target.value)}
                placeholder="/absolute/path/to/obsidian-vault"
                disabled={busy}
              />
              <p className="review-source">
                Parses every .md note under this folder and proposes topics/dependencies from its [[wikilinks]] --
                same review-before-apply flow as ingest.
              </p>
              {error ? <p className="modal__error">{error}</p> : null}
              <div className="modal__actions">
                <button type="button" className="modal__btn modal__btn--ghost" onClick={handleClose} disabled={busy}>
                  Cancel
                </button>
                <button type="submit" className="modal__btn modal__btn--primary" disabled={busy || !vaultPath.trim()}>
                  {busy ? "Importing…" : "Import vault"}
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
              {auditReport.status === "partial" || auditReport.semantic_analysis === "unavailable" ? (
                <p className="audit-degraded" role="status">
                  Degraded mode: structural findings only. Semantic AI analysis was unavailable
                  {auditReport.semantic_error ? ` (${auditReport.semantic_error})` : ""}.
                </p>
              ) : null}
              <p className="review-source">
                {auditReport.total_topics} topic(s) analyzed, {auditReport.findings.length} finding(s)
                {auditReport.semantic_analysis === "unavailable" ? " (structural only)" : ""}.
              </p>
              {auditReport.findings.length === 0 ? (
                <p className="sidebar__muted">
                  {auditReport.semantic_analysis === "unavailable"
                    ? "No structural issues found. Semantic analysis was unavailable."
                    : "No issues found."}
                </p>
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
            <ProposalDetails
              proposal={proposal}
              nodes={nodes}
              busy={busy}
              error={error}
              onApply={() => void handleApply()}
              onDiscard={() => void handleDiscard()}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}
