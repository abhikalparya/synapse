import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { AiOperationsModal } from "./components/AiOperationsModal";
import { GraphSearchBar } from "./components/GraphSearchBar";
import { KnowledgeGraph } from "./components/KnowledgeGraph";
import { NodeDetailsPanel } from "./components/NodeDetailsPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { Sidebar } from "./components/Sidebar";
import { linkKey, prepareGraphData } from "./graphUtils";
import type { ApplyResponse, GraphData, GraphNode, PathResponse, RollbackResponse, StatsResponse, Zone } from "./types";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const text = await res.text();
    let message = text || res.statusText;
    try {
      const j = JSON.parse(text) as { detail?: unknown };
      if (typeof j.detail === "string") message = j.detail;
    } catch {
      // not JSON -- fall back to raw text
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

const emptyGraph = (): GraphData => ({ nodes: [], links: [] });
const NO_IDS: readonly string[] = [];
const NO_USED_IDS = new Set<string>();

export default function App() {
  const [graphData, setGraphData] = useState<GraphData>(() => prepareGraphData(emptyGraph()));
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [statsLoading, setStatsLoading] = useState(true);
  const [graphError, setGraphError] = useState<string | null>(null);

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [reheatToken, setReheatToken] = useState(0);
  const [pathNodeIds, setPathNodeIds] = useState<Set<string>>(new Set());
  const [pathLinkKeys, setPathLinkKeys] = useState<Set<string>>(new Set());
  const [aiModalOpen, setAiModalOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [undoBusy, setUndoBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [zones, setZones] = useState<Zone[]>([]);

  const graphAreaRef = useRef<HTMLDivElement>(null);
  const focusCameraNonceRef = useRef(0);
  const [focusCameraRequest, setFocusCameraRequest] = useState<{ nodeId: string; nonce: number } | null>(null);
  const [graphSize, setGraphSize] = useState({ w: 400, h: 400 });

  const layoutSnapshotRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const graphDataRef = useRef(graphData);
  graphDataRef.current = graphData;

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

  const refreshZones = useCallback(async () => {
    try {
      setZones(await fetchJson<Zone[]>("/zones"));
    } catch {
      setZones([]);
    }
  }, []);

  useEffect(() => {
    void refreshGraph({ silent: false, preserveLayout: false });
    void refreshStats();
    void refreshZones();
  }, [refreshGraph, refreshStats, refreshZones]);

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

  useEffect(() => {
    const id = selectedNode?.id;
    if (!id) {
      setPathNodeIds(new Set());
      setPathLinkKeys(new Set());
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetchJson<PathResponse>(`/graph/path?target=${encodeURIComponent(id)}`);
        if (cancelled) return;
        setPathNodeIds(new Set(res.chain.map((c) => c.id)));
        setPathLinkKeys(new Set(res.edges.map((e) => linkKey(e.source, e.target))));
      } catch {
        if (cancelled) return;
        setPathNodeIds(new Set());
        setPathLinkKeys(new Set());
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedNode?.id]);

  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(t);
  }, [toast]);

  const handleApplied = useCallback(
    async (result: ApplyResponse) => {
      await refreshGraph({ silent: true, preserveLayout: true });
      await refreshStats();
      setToast(`Applied: ${result.created_topics.length} topic(s), ${result.created_dependencies.length} dependency(ies)`);
    },
    [refreshGraph, refreshStats],
  );

  const handleTopicChanged = useCallback(async () => {
    await refreshGraph({ silent: true, preserveLayout: true });
    await refreshStats();
  }, [refreshGraph, refreshStats]);

  const handleCreateZone = useCallback(
    async (label: string, color: string) => {
      try {
        await fetchJson("/zones", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label, color }),
        });
        await refreshZones();
      } catch (e) {
        setToast(e instanceof Error ? e.message : "Failed to create zone");
      }
    },
    [refreshZones],
  );

  const handleUndoLastChange = useCallback(async () => {
    setUndoBusy(true);
    try {
      const res = await fetchJson<RollbackResponse>("/rollback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      await refreshGraph({ silent: true, preserveLayout: false });
      await refreshStats();
      setSelectedNode(null);
      setToast(`Rolled back to snapshot ${res.snapshot_id}`);
    } catch (e) {
      setToast(e instanceof Error ? e.message : "Nothing to undo");
    } finally {
      setUndoBusy(false);
    }
  }, [refreshGraph, refreshStats]);

  const resolveNodeById = useCallback(
    (id: string) => graphData.nodes.find((n) => n.id === id) ?? null,
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

  const handlePickNode = useCallback(
    (id: string) => {
      const n = resolveNodeById(id);
      if (n) {
        setSelectedNode(n);
        bumpFocusCamera(n.id);
      } else {
        setSelectedNode(null);
      }
    },
    [bumpFocusCamera, resolveNodeById],
  );

  const showGraph = graphSize.w > 0 && graphSize.h > 0;
  const hasNodes = graphData.nodes.length > 0;
  const showKnowledgeGraph = showGraph && hasNodes;
  const emptyBrain = !graphLoading && !hasNodes && !graphError;
  const canvasLoadingEmpty = graphLoading && !hasNodes && !graphError;

  return (
    <div className="app">
      {toast ? (
        <div className="app__toast" role="status">
          {toast}
        </div>
      ) : null}
      <Sidebar
        stats={stats}
        loading={statsLoading}
        onPickNode={handlePickNode}
        onOpenAiOperations={() => setAiModalOpen(true)}
        onOpenSettings={() => setSettingsOpen(true)}
        onUndoLastChange={() => void handleUndoLastChange()}
        undoBusy={undoBusy}
        zones={zones}
        onCreateZone={handleCreateZone}
      />

      <main className="app__main">
        <header className="app__header">
          <div className="app__header-lead">
            <h1 className="app__title">Dependency graph</h1>
            <p className="app__subtitle">Topics · directed prerequisites</p>
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
              Syncing dependency graph…
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
              <p className="app__empty-message">No topics yet. Create some via the API to see the graph.</p>
            </div>
          ) : null}
          {showKnowledgeGraph ? (
            <KnowledgeGraph
              data={graphData}
              width={graphSize.w}
              height={graphSize.h}
              selectedId={selectedNode?.id ?? null}
              onSelectNode={setSelectedNode}
              queryUsedIds={NO_USED_IDS}
              queryUpdatedId={null}
              reheatToken={reheatToken}
              onLayoutSnapshot={captureLayout}
              birthNodeIds={NO_IDS}
              queryPulseIds={NO_IDS}
              focusCameraRequest={focusCameraRequest}
              pathNodeIds={pathNodeIds}
              pathLinkKeys={pathLinkKeys}
              zones={zones}
            />
          ) : null}
        </div>
      </main>

      <NodeDetailsPanel
        graphData={graphData}
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onNavigateToNode={navigateToNode}
        onTopicChanged={() => void handleTopicChanged()}
        zones={zones}
      />

      <AiOperationsModal
        open={aiModalOpen}
        onClose={() => setAiModalOpen(false)}
        nodes={graphData.nodes}
        onApplied={(result) => void handleApplied(result)}
      />

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
