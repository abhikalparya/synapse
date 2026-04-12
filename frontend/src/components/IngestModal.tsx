import { useCallback, useEffect, useId, useRef, useState, type FormEvent } from "react";
import { FileTypeIcon, fileExt } from "./FileTypeIcon";

type Props = {
  open: boolean;
  onClose: () => void;
  /** Refreshes graph & stats after a successful ingest */
  onSuccess: () => void | Promise<void>;
};

type GenerateResponse = {
  processed: number;
  created: string[];
  errors: { source: string; detail: string }[];
};

type BatchIngestItem = {
  filename: string;
  status: "ok" | "warning" | "error";
  path?: string | null;
  saved_filename?: string | null;
  warnings?: string[];
  file_type?: string | null;
  detail?: string | null;
};

type BatchIngestResponse = {
  items: BatchIngestItem[];
};

const ACCEPT =
  ".txt,.md,.pdf,.docx,text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

const ACCEPT_EXT = new Set([".txt", ".md", ".pdf", ".docx"]);

function fileKey(f: File): string {
  return `${f.name}:${f.size}:${f.lastModified}`;
}

function formatIngestError(status: number, bodyText: string): string {
  try {
    const j = JSON.parse(bodyText) as { detail?: unknown };
    const d = j.detail;
    if (typeof d === "string") {
      if (status === 415) return `Unsupported format. ${d}`;
      if (status === 422) return `Extraction failed. ${d}`;
      return d;
    }
    if (Array.isArray(d)) {
      return d
        .map((x) =>
          typeof x === "object" && x !== null && "msg" in x ? String((x as { msg: string }).msg) : JSON.stringify(x),
        )
        .join("; ");
    }
  } catch {
    /* ignore */
  }
  const t = bodyText.trim();
  return t || `Request failed (${status})`;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const text = await res.text();
  if (!res.ok) {
    throw new Error(formatIngestError(res.status, text));
  }
  return JSON.parse(text) as T;
}

async function postGenerateFromRaw(filenames: string[]): Promise<GenerateResponse> {
  const unique = [...new Set(filenames.map((s) => s.trim()).filter(Boolean))];
  if (!unique.length) {
    return fetchJson<GenerateResponse>("/generate", { method: "POST" });
  }
  return fetchJson<GenerateResponse>("/generate/from-raw", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filenames: unique }),
  });
}

type IngestUploadJson = {
  status: string;
  filename?: string | null;
};

async function postIngestMultipart(file: File): Promise<IngestUploadJson> {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch("/ingest/upload", {
    method: "POST",
    body: form,
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(formatIngestError(res.status, text));
  }
  try {
    return JSON.parse(text) as IngestUploadJson;
  } catch {
    throw new Error("Invalid response from server");
  }
}

async function postIngestMultipartBatch(files: File[]): Promise<BatchIngestResponse> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f, f.name);
  }
  const res = await fetch("/ingest/upload/batch", {
    method: "POST",
    body: form,
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(formatIngestError(res.status, text));
  }
  try {
    return JSON.parse(text) as BatchIngestResponse;
  } catch {
    throw new Error("Invalid response from server");
  }
}

export function IngestModal({ open, onClose, onSuccess }: Props) {
  const idBase = useId();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);
  const [text, setText] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [runGenerate, setRunGenerate] = useState(true);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<"idle" | "parsing" | "generating" | "saving">("idle");
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    if (!open) {
      setDragOver(false);
    }
  }, [open]);

  const reset = useCallback(() => {
    setText("");
    setSelectedFiles([]);
    setError(null);
    setToast(null);
    setDragOver(false);
    setPhase("idle");
    setRunGenerate(true);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const handleClose = useCallback(() => {
    if (busy) return;
    reset();
    onClose();
  }, [busy, onClose, reset]);

  const mergeValidatedFiles = useCallback((existing: File[], incoming: File[]) => {
    const rejected: string[] = [];
    const seen = new Set(existing.map(fileKey));
    const next = [...existing];
    for (const f of incoming) {
      const ext = fileExt(f.name);
      if (!ACCEPT_EXT.has(ext)) {
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

  const addFilesFromList = useCallback(
    (list: FileList | File[] | null) => {
      if (!list?.length) return;
      const incoming = Array.from(list as FileList);
      setSelectedFiles((prev) => {
        const { next, rejected } = mergeValidatedFiles(prev, incoming);
        requestAnimationFrame(() => {
          if (rejected.length) {
            setError(
              `Skipped unsupported type: ${rejected.slice(0, 4).join(", ")}${rejected.length > 4 ? "…" : ""} (.txt, .md, .pdf, .docx only)`,
            );
          } else {
            setError(null);
          }
        });
        return next;
      });
      if (fileInputRef.current) fileInputRef.current.value = "";
    },
    [mergeValidatedFiles],
  );

  const removeFileAt = useCallback((index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setError(null);
  }, []);

  const clearAllFiles = useCallback(() => {
    setSelectedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
    setError(null);
  }, []);

  const openFilePicker = useCallback(() => {
    if (busy) return;
    fileInputRef.current?.click();
  }, [busy]);

  const setDropHighlight = useCallback((on: boolean) => {
    setDragOver(on);
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = text.trim();
    if (busy) return;
    if (!selectedFiles.length && !trimmed) return;

    setBusy(true);
    setError(null);
    setToast(null);
    setPhase(selectedFiles.length ? "parsing" : "saving");

    let successToast: string | null = null;

    try {
      if (selectedFiles.length) {
        if (selectedFiles.length === 1) {
          const up = await postIngestMultipart(selectedFiles[0]);
          if (runGenerate) {
            setPhase("generating");
            if (up.filename) {
              await postGenerateFromRaw([up.filename]);
            } else {
              await fetchJson<GenerateResponse>("/generate", { method: "POST" });
            }
          }
        } else {
          const batch = await postIngestMultipartBatch(selectedFiles);
          const ok = batch.items.filter((i) => i.status === "ok" || i.status === "warning").length;
          const fail = batch.items.filter((i) => i.status === "error").length;
          if (ok === 0) {
            const msg = batch.items.map((i) => `${i.filename}: ${i.detail ?? "failed"}`).join("; ");
            throw new Error(msg || "No files could be saved");
          }
          if (fail > 0) {
            successToast = `Saved ${ok} of ${batch.items.length} files (${fail} failed).`;
          } else {
            successToast = `Knowledge added to your brain (${selectedFiles.length} files)`;
          }
          if (runGenerate) {
            setPhase("generating");
            const savedNames = batch.items
              .filter((i) => i.status === "ok" || i.status === "warning")
              .map((i) => i.saved_filename)
              .filter((x): x is string => Boolean(x));
            if (savedNames.length) {
              await postGenerateFromRaw(savedNames);
            } else {
              await fetchJson<GenerateResponse>("/generate", { method: "POST" });
            }
          }
        }
      } else {
        const pasted = await fetchJson<IngestUploadJson>("/ingest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: trimmed }),
        });
        if (runGenerate) {
          setPhase("generating");
          if (pasted.filename) {
            await postGenerateFromRaw([pasted.filename]);
          } else {
            await fetchJson<GenerateResponse>("/generate", { method: "POST" });
          }
        }
      }

      setToast(
        successToast ??
          (selectedFiles.length > 1
            ? `Knowledge added to your brain (${selectedFiles.length} files)`
            : "Knowledge added to your brain"),
      );
      await onSuccess();
      window.setTimeout(() => {
        reset();
        onClose();
      }, 1600);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setPhase("idle");
    } finally {
      setBusy(false);
      setPhase("idle");
    }
  }

  if (!open) return null;

  const canSubmit = Boolean(selectedFiles.length || text.trim());
  const loadingLabel =
    phase === "generating"
      ? "Generating wiki pages…"
      : phase === "parsing"
        ? selectedFiles.length > 1
          ? `Uploading ${selectedFiles.length} documents…`
          : "Parsing document…"
        : phase === "saving"
          ? "Saving note…"
          : "";

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(ev) => ev.target === ev.currentTarget && handleClose()}>
      <div
        className="modal modal--ingest"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${idBase}-title`}
        aria-busy={busy}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="modal__header">
          <h2 id={`${idBase}-title`} className="modal__title">
            Add Knowledge
          </h2>
          <button type="button" className="modal__close" onClick={handleClose} disabled={busy} aria-label="Close">
            ×
          </button>
        </div>

        <form className="modal__body" onSubmit={handleSubmit}>
          <label className="modal__label" id={`${idBase}-upload-label`}>
            Files
          </label>

          <input
            ref={fileInputRef}
            id={`${idBase}-file`}
            type="file"
            className="visually-hidden"
            accept={ACCEPT}
            multiple
            disabled={busy}
            tabIndex={-1}
            aria-labelledby={`${idBase}-upload-label`}
            onChange={(e) => addFilesFromList(e.target.files)}
          />

          <div
            ref={dropRef}
            className={`ingest-drop ${dragOver ? "ingest-drop--active" : ""} ${selectedFiles.length ? "ingest-drop--has-file" : ""}`}
            role="group"
            aria-label="Drop documents here or browse to add files"
            onDragEnter={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDropHighlight(true);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDropHighlight(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              e.stopPropagation();
              if (!dropRef.current?.contains(e.relatedTarget as Node)) setDropHighlight(false);
            }}
            onDrop={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setDropHighlight(false);
              addFilesFromList(e.dataTransfer.files);
            }}
          >
            {selectedFiles.length === 0 ? (
              <div
                className="ingest-drop__empty"
                role="button"
                tabIndex={0}
                onClick={() => !busy && openFilePicker()}
                onKeyDown={(e) => {
                  if (e.key !== "Enter" && e.key !== " ") return;
                  e.preventDefault();
                  if (!busy) openFilePicker();
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
                    openFilePicker();
                  }}
                >
                  Browse files
                </button>
              </div>
            ) : (
              <div className="ingest-drop__multi" onClick={(e) => e.stopPropagation()}>
                <ul className="ingest-file-list" aria-label="Selected files">
                  {selectedFiles.map((f, i) => (
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
                        onClick={() => removeFileAt(i)}
                        disabled={busy}
                        aria-label={`Remove ${f.name}`}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
                <div className="ingest-drop__multi-actions">
                  <button type="button" className="ingest-drop__change" onClick={openFilePicker} disabled={busy}>
                    Add more files
                  </button>
                  <button type="button" className="ingest-drop__clear-all" onClick={clearAllFiles} disabled={busy}>
                    Clear all
                  </button>
                </div>
              </div>
            )}
          </div>

          <label className="modal__label" htmlFor={`${idBase}-ta`}>
            Or paste a note
          </label>
          <textarea
            id={`${idBase}-ta`}
            className="modal__textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste or type plain text…"
            rows={5}
            disabled={busy}
          />

          <label className="modal__check">
            <input type="checkbox" checked={runGenerate} onChange={(e) => setRunGenerate(e.target.checked)} disabled={busy} />
            <span>Generate wiki after upload</span>
          </label>

          {selectedFiles.length > 0 && text.trim() ? (
            <p className="modal__hint">Files are selected — Save uploads files only (not the note text).</p>
          ) : null}

          {busy && loadingLabel ? (
            <div className="ingest-status" role="status">
              <span className="ingest-status__spinner" aria-hidden />
              <span className="ingest-status__text">{loadingLabel}</span>
            </div>
          ) : null}

          {error ? <p className="modal__error">{error}</p> : null}
          {toast ? <p className="modal__toast modal__toast--success">{toast}</p> : null}

          <div className="modal__actions">
            <button type="button" className="modal__btn modal__btn--ghost" onClick={handleClose} disabled={busy}>
              Cancel
            </button>
            <button type="submit" className="modal__btn modal__btn--primary" disabled={busy || !canSubmit}>
              {busy ? "Working…" : "Save to brain"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
