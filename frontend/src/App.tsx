import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { GraphSearchBar } from "./components/GraphSearchBar";
import { IngestModal } from "./components/IngestModal";
import { KnowledgeGraph } from "./components/KnowledgeGraph";
import { NodeDetailsPanel } from "./components/NodeDetailsPanel";
import { QueryBar } from "./components/QueryBar";
import { Sidebar } from "./components/Sidebar";
import { prepareGraphData } from "./graphUtils";
import type { GraphData, GraphNode, QueryResponse, RefactorResponse, StatsResponse } from "./types";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

const emptyGraph = (): GraphData => ({ nodes: [], links: [] });

export default function App() {
  const [graphData, setGraphData] = useState<GraphData>(() => prepareGraphData(emptyGraph()));
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [statsLoading, setStatsLoading] = useState(true);
  const [graphError, setGraphError] = useState<string | null>(null);

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryAnswerEpoch, setQueryAnswerEpoch] = useState(0);
  const [queryPulseEpoch, setQueryPulseEpoch] = useState(0);
  const [queryPulseIds, setQueryPulseIds] = useState<readonly string[]>([]);
  const [reheatToken, setReheatToken] = useState(0);
  const [ingestOpen, setIngestOpen] = useState(false);
  const [birthEpoch, setBirthEpoch] = useState(0);
  const [birthNodeIds, setBirthNodeIds] = useState<string[]>([]);
  const [refactorLoading, setRefactorLoading] = useState(false);
  const [refactorError, setRefactorError] = useState<string | null>(null);
  const [refactorResult, setRefactorResult] = useState<RefactorResponse | null>(null);
  const [successToast, setSuccessToast] = useState<string | null>(null);
  const [graphUpdatePulse, setGraphUpdatePulse] = useState(0);

  const queueNodeBirth = useCallback((ids: string[]) => {
    const unique = [...new Set(ids.map((s) => s.trim()).filter(Boolean))];
    if (!unique.length) return;
    setBirthNodeIds(unique);
    setBirthEpoch((e) => e + 1);
  }, []);

  const graphAreaRef = useRef<HTMLDivElement>(null);
  const selectedNodeRef = useRef<GraphNode | null>(null);
  const focusCameraNonceRef = useRef(0);
  const [focusCameraRequest, setFocusCameraRequest] = useState<{ nodeId: string; nonce: number } | null>(null);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);
  const [graphFitAfterQueryEpoch, setGraphFitAfterQueryEpoch] = useState(0);
  const [graphSize, setGraphSize] = useState({ w: 400, h: 400 });

  const layoutSnapshotRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const graphDataRef = useRef(graphData);
  graphDataRef.current = graphData;
  selectedNodeRef.current = selectedNode;

  const captureLayout = useCallback(() => {
    for (const n of graphDataRef.current.nodes) {
      if (typeof n.x === "number" && typeof n.y === "number") {
        layoutSnapshotRef.current.set(n.id, { x: n.x, y: n.y });
      }
    }
  }, []);

  useEffect(() => {
    setSelectedNode((prev) => {
      if (!prev) return prev;
      const next = graphData.nodes.find((n) => n.id === prev.id);
      return next ?? prev;
    });
  }, [graphData]);

  const queryUsedIds = useMemo(
    () => new Set(queryResult?.used_nodes ?? []),
    [queryResult?.used_nodes],
  );
  const queryUpdatedId = queryResult?.updated_node ?? null;

  const refreshGraph = useCallback(async (opts?: { silent?: boolean; preserveLayout?: boolean }) => {
    const silent = opts?.silent ?? false;
    const preserve = opts?.preserveLayout ?? false;
    if (!silent) setGraphLoading(true);
    setGraphError(null);
    try {
      if (!preserve) {
        layoutSnapshotRef.current = new Map();
      }
      const g = await fetchJson<GraphData>("/graph");
      const prepared = prepareGraphData(g, {
        snapshot: preserve ? layoutSnapshotRef.current : undefined,
        maxLinksPerNode: 14,
      });
      setGraphData(prepared);
      setReheatToken((x) => x + 1);
    } catch (e) {
      setGraphError(e instanceof Error ? e.message : "Failed to load graph");
    } finally {
      if (!silent) setGraphLoading(false);
    }
  }, []);

  const refreshStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      const s = await fetchJson<StatsResponse>("/stats");
      setStats(s);
    } catch {
      setStats(null);
    } finally {
      setStatsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshGraph({ silent: false, preserveLayout: false });
    void refreshStats();
  }, [refreshGraph, refreshStats]);

  useEffect(() => {
    const el = graphAreaRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setGraphSize({ w: el.clientWidth, h: el.clientHeight });
    });
    ro.observe(el);
    setGraphSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const resolveNodeByTitle = useCallback(
    (title: string) => graphData.nodes.find((n) => n.id === title) ?? null,
    [graphData.nodes],
  );

  const bumpFocusCamera = useCallback((nodeId: string) => {
    focusCameraNonceRef.current += 1;
    setFocusCameraRequest({ nodeId, nonce: focusCameraNonceRef.current });
  }, []);

  const navigateToNode = useCallback(
    (node: GraphNode) => {
      setSelectedNode(node);
      bumpFocusCamera(node.id);
    },
    [bumpFocusCamera],
  );

  const handleQuery = useCallback(
    async (text: string) => {
      setQueryError(null);
      setQueryLoading(true);
      captureLayout();
      try {
        const body = await fetchJson<QueryResponse>("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: text }),
        });
        setQueryResult(body);
        setQueryAnswerEpoch((n) => n + 1);
        const trimmed = text.trim();
        if (trimmed) {
          setQueryHistory((h) => [trimmed, ...h.filter((x) => x !== trimmed)].slice(0, 5));
        }
        await refreshGraph({ silent: true, preserveLayout: true });
        if (selectedNodeRef.current) {
          setGraphFitAfterQueryEpoch((n) => n + 1);
        }
        const pulseIds = [
          ...new Set(
            [...body.used_nodes, ...(body.updated_node ? [body.updated_node] : [])].map((s) => s.trim()).filter(Boolean),
          ),
        ];
        setQueryPulseIds(pulseIds);
        setQueryPulseEpoch((n) => n + 1);
        if (body.updated_node) {
          queueNodeBirth([body.updated_node]);
        }
        await refreshStats();
      } catch (e) {
        setQueryError(e instanceof Error ? e.message : "Query failed");
        throw e;
      } finally {
        setQueryLoading(false);
      }
    },
    [captureLayout, queueNodeBirth, refreshGraph, refreshStats],
  );

  const handlePickTitle = useCallback(
    (title: string) => {
      const n = resolveNodeByTitle(title);
      if (n) {
        setSelectedNode(n);
        bumpFocusCamera(n.id);
      } else {
        setSelectedNode(null);
      }
    },
    [bumpFocusCamera, resolveNodeByTitle],
  );

  useEffect(() => {
    if (!successToast) return;
    const t = window.setTimeout(() => setSuccessToast(null), 4000);
    return () => window.clearTimeout(t);
  }, [successToast]);

  const handleRefactor = useCallback(async () => {
    setRefactorError(null);
    setRefactorLoading(true);
    captureLayout();
    try {
      const body = await fetchJson<RefactorResponse>("/refactor", { method: "POST" });
      setRefactorResult(body);
      setSuccessToast("Knowledge improved successfully");
      await refreshGraph({ silent: true, preserveLayout: true });
      await refreshStats();
      setGraphUpdatePulse((n) => n + 1);
    } catch (e) {
      setRefactorError(e instanceof Error ? e.message : "Refactor failed");
    } finally {
      setRefactorLoading(false);
    }
  }, [captureLayout, refreshGraph, refreshStats]);

  const handleIngestSuccess = useCallback(async () => {
    const idsBefore = new Set(graphDataRef.current.nodes.map((n) => n.id));
    captureLayout();
    await refreshGraph({ silent: true, preserveLayout: true });
    await refreshStats();
    const idsAfter = graphDataRef.current.nodes.map((n) => n.id);
    const newIds = idsAfter.filter((id) => !idsBefore.has(id));
    if (newIds.length) {
      queueNodeBirth(newIds);
    }
  }, [captureLayout, queueNodeBirth, refreshGraph, refreshStats]);

  const showGraph = graphSize.w > 0 && graphSize.h > 0;
  const hasNodes = graphData.nodes.length > 0;
  const showKnowledgeGraph = showGraph && hasNodes;
  const emptyBrain = !graphLoading && !hasNodes && !graphError;
  const canvasLoadingEmpty = graphLoading && !hasNodes && !graphError;

  return (
    <div className="app">
      {successToast ? (
        <div className="app__toast" role="status">
          {successToast}
        </div>
      ) : null}
      <Sidebar
        stats={stats}
        loading={statsLoading}
        onPickTitle={handlePickTitle}
        onAddNote={() => setIngestOpen(true)}
        onRefactor={() => void handleRefactor()}
        refactorLoading={refactorLoading}
        refactorError={refactorError}
        refactorResult={refactorResult}
      />

      <main className="app__main">
        <header className="app__header">
          <div className="app__header-lead">
            <h1 className="app__title">Knowledge graph</h1>
            <p className="app__subtitle">Live structure · query-evolving memory</p>
          </div>
          {hasNodes ? (
            <GraphSearchBar nodes={graphData.nodes} onNavigateToNode={navigateToNode} />
          ) : null}
          {graphLoading ? <span className="badge badge--pulse">Syncing…</span> : null}
        </header>

        <div className="app__canvas" ref={graphAreaRef}>
          {graphError ? <div className="app__error">{graphError}</div> : null}
          {canvasLoadingEmpty ? (
            <div className="app__canvas-loading" aria-live="polite">
              <span className="app__canvas-loading__dot" aria-hidden />
              Syncing knowledge graph…
            </div>
          ) : null}
          {emptyBrain ? (
            <div className="app__empty app__empty--brain">
              <div className="app__empty-icon" aria-hidden>
                <svg viewBox="0 0 64 64" width="56" height="56" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <circle cx="20" cy="22" r="5" stroke="currentColor" strokeWidth="2" opacity="0.9" />
                  <circle cx="44" cy="18" r="4" stroke="currentColor" strokeWidth="2" opacity="0.75" />
                  <circle cx="38" cy="42" r="5" stroke="currentColor" strokeWidth="2" opacity="0.85" />
                  <circle cx="14" cy="44" r="3.5" stroke="currentColor" strokeWidth="2" opacity="0.65" />
                  <path
                    d="M23 24c6 4 10 2 14-4M24 28c4 8 8 10 12 8M20 40c6-2 10 0 14 4"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    opacity="0.45"
                  />
                </svg>
              </div>
              <p className="app__empty-message">Your second brain is empty. Add knowledge to begin.</p>
              <button type="button" className="app__empty-cta" onClick={() => setIngestOpen(true)}>
                Add Knowledge
              </button>
            </div>
          ) : null}
          {showKnowledgeGraph ? (
            <KnowledgeGraph
              data={graphData}
              width={graphSize.w}
              height={graphSize.h}
              selectedId={selectedNode?.id ?? null}
              onSelectNode={setSelectedNode}
              queryUsedIds={queryUsedIds}
              queryUpdatedId={queryUpdatedId}
              reheatToken={reheatToken}
              onLayoutSnapshot={captureLayout}
              birthEpoch={birthEpoch}
              birthNodeIds={birthNodeIds}
              queryPulseEpoch={queryPulseEpoch}
              queryPulseIds={queryPulseIds}
              fitAfterQueryEpoch={graphFitAfterQueryEpoch}
              focusCameraRequest={focusCameraRequest}
              structuralUpdatePulse={graphUpdatePulse}
            />
          ) : null}
        </div>

        <div className="app__bottom">
          <QueryBar
            answer={queryResult?.answer ?? null}
            loading={queryLoading}
            error={queryError}
            answerEpoch={queryAnswerEpoch}
            queryHistory={queryHistory}
            onReplayHistory={(q) => {
              void handleQuery(q);
            }}
            onSubmit={handleQuery}
            onDismissError={() => setQueryError(null)}
          />
        </div>
      </main>

      <NodeDetailsPanel
        graphData={graphData}
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onNavigateToNode={navigateToNode}
      />

      <IngestModal open={ingestOpen} onClose={() => setIngestOpen(false)} onSuccess={handleIngestSuccess} />
    </div>
  );
}
