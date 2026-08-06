import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";
import { forceCenter, forceCollide } from "d3-force";
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { MutableRefObject } from "react";
import { BIRTH_TOTAL_MS, birthVisual } from "../graphBirth";
import { QUERY_PULSE_TOTAL_MS, queryPulseScale } from "../graphQueryPulse";
import type { GraphData, GraphLink, GraphNode } from "../types";
import { groupColor, linkKey, neighborSets, nodeRadius } from "../graphUtils";

const NO_BIRTH: readonly string[] = [];
const NO_PATH_IDS: ReadonlySet<string> = new Set();
const PATH_HIGHLIGHT_COLOR = "#4fd1ff";

let colorParseCtx: CanvasRenderingContext2D | null = null;

function cssColorToRgbString(css: string): string {
  if (typeof document === "undefined") return "rgb(120, 130, 190)";
  if (!colorParseCtx) {
    const c = document.createElement("canvas");
    colorParseCtx = c.getContext("2d");
  }
  if (!colorParseCtx) return "rgb(120, 130, 190)";
  colorParseCtx.fillStyle = "#000";
  colorParseCtx.fillStyle = css;
  const out = colorParseCtx.fillStyle as string;
  return typeof out === "string" ? out : "rgb(120, 130, 190)";
}

/** Multiply alpha of an rgb/rgba color string (browser-normalized). */
function fadeRgbColor(rgbString: string, opacityMult: number): string {
  const m = rgbString.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)/);
  if (!m) return `rgba(120, 130, 190, ${0.72 * opacityMult})`;
  const r = m[1];
  const g = m[2];
  const b = m[3];
  const a0 = m[4] !== undefined ? Number(m[4]) : 1;
  const a = Math.min(1, Math.max(0, a0 * opacityMult));
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

type GraphAccessorCtx = {
  data: GraphData;
  dimmingFocusId: string | null;
  focusNeighborNodes: Set<string>;
  focusLinkKeys: Set<string>;
  hoverDimRef: MutableRefObject<number>;
  /** When true, only edges between the focus node and its neighbors are shown. */
  restrictLinksToFocus: boolean;
  /** Node ids along the currently highlighted prerequisite chain (empty when none). */
  pathNodeIds: ReadonlySet<string>;
  /** Directed edge keys (``linkKey(source, target)``) along the highlighted chain. */
  pathLinkKeys: ReadonlySet<string>;
};

type PaintCtx = {
  /** Hover or selection — glow / interaction ring */
  interactionFocusId: string | null;
  /** Neighbor dimming; null when query highlights active → full graph brightness */
  dimmingFocusId: string | null;
  selectedId: string | null;
  queryUsedIds: Set<string>;
  queryUpdatedId: string | null;
  data: GraphData;
  birthStartsRef: MutableRefObject<Map<string, number>>;
  queryPulseStartsRef: MutableRefObject<Map<string, number>>;
  focusNeighborNodes: Set<string>;
  hoverDimRef: MutableRefObject<number>;
};

type Props = {
  data: GraphData;
  width: number;
  height: number;
  selectedId: string | null;
  onSelectNode: (node: GraphNode | null) => void;
  queryUsedIds: Set<string>;
  queryUpdatedId: string | null;
  reheatToken: number;
  onLayoutSnapshot?: () => void;
  /** Increment to (re)play birth animation for `birthNodeIds` */
  birthEpoch?: number;
  /** Node titles to animate (e.g. `updated_node` or newly ingested ids) */
  birthNodeIds?: readonly string[];
  /** Increment after each successful query to replay impact pulse */
  queryPulseEpoch?: number;
  /** Node ids to pulse (used + updated); same render as epoch bump */
  queryPulseIds?: readonly string[];
  /** Increment after query with a node selected to zoom out to full graph */
  fitAfterQueryEpoch?: number;
  /** Bump `nonce` to smoothly center/zoom on `nodeId` (e.g. search / sidebar). */
  focusCameraRequest?: { nodeId: string; nonce: number } | null;
  /** Increment after bulk graph refresh (e.g. refactor) for a brief visual soften. */
  structuralUpdatePulse?: number;
  /** Node ids along the currently highlighted prerequisite chain (from `/graph/path`). */
  pathNodeIds?: ReadonlySet<string>;
  /** Directed edge keys (``linkKey(source, target)``) along the highlighted chain. */
  pathLinkKeys?: ReadonlySet<string>;
};

function wantGlowBefore(n: GraphNode, p: PaintCtx): boolean {
  const id = n.id;
  if (p.queryUsedIds.has(id)) return false;
  if (p.queryUpdatedId && id === p.queryUpdatedId) return false;
  return p.interactionFocusId === id;
}

function nodeCanvasMode(n: GraphNode, p: PaintCtx): "before" | "after" {
  return wantGlowBefore(n, p) ? "before" : "after";
}

function paintNodeCanvas(
  node: GraphNode,
  ctx: CanvasRenderingContext2D,
  globalScale: number,
  p: PaintCtx,
) {
  const id = node.id;
  const baseR = nodeRadius(node, 3.1);
  if (node.x === undefined || node.y === undefined) return;

  const used = p.queryUsedIds.has(id);
  const updated = Boolean(p.queryUpdatedId && id === p.queryUpdatedId);

  const birthStart = p.birthStartsRef.current.get(id);
  const birth =
    birthStart !== undefined ? birthVisual(performance.now() - birthStart) : null;
  const pulseStart = p.queryPulseStartsRef.current.get(id);
  const pulseMs = pulseStart !== undefined ? performance.now() - pulseStart : -1;
  const pulseS = pulseMs >= 0 ? queryPulseScale(pulseMs) : 1;
  const radius = (birth ? baseR * birth.scale : baseR) * pulseS;

  if (wantGlowBefore(node, p)) {
    let blur = 7;
    let color = "rgba(129, 161, 255, 0.35)";
    if (p.selectedId === id) {
      blur = 11;
      color = "rgba(180, 170, 255, 0.45)";
    }
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = blur / globalScale;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius + 1.2, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(0,0,0,0)";
    ctx.fill();
    ctx.restore();

    if (p.selectedId === id) {
      ctx.save();
      ctx.strokeStyle = "rgba(226, 220, 255, 0.5)";
      ctx.lineWidth = 1.25 / globalScale;
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius + 2.1, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
    } else if (p.interactionFocusId === id) {
      ctx.save();
      ctx.strokeStyle = "rgba(186, 198, 255, 0.32)";
      ctx.lineWidth = 1.05 / globalScale;
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius + 1.75, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
    }
    return;
  }

  const dim = p.hoverDimRef.current;
  if (!used && !updated && p.dimmingFocusId && !p.focusNeighborNodes.has(id) && dim > 0.004) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius + 2.2, 0, 2 * Math.PI);
    ctx.fillStyle = `rgba(8, 11, 20, ${0.08 + dim * 0.52})`;
    ctx.fill();
  }

  if (birth && birth.tintAlpha > 0.02) {
    ctx.save();
    ctx.globalAlpha = Math.min(1, birth.tintAlpha);
    ctx.shadowColor = "rgba(52, 255, 160, 0.55)";
    ctx.shadowBlur = (12 + 8 * birth.tintAlpha) / globalScale;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = "#39ff9e";
    ctx.fill();
    ctx.restore();
  }

  if (updated && (!birth || birth.tintAlpha < 0.14)) {
    ctx.save();
    ctx.shadowColor = "rgba(52, 211, 153, 0.4)";
    ctx.shadowBlur = 10 / globalScale;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = "#34f0a8";
    ctx.fill();
    ctx.restore();
    return;
  }

  if (used) {
    ctx.save();
    ctx.shadowColor = "rgba(245, 200, 120, 0.35)";
    ctx.shadowBlur = 8 / globalScale;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = "#e8c170";
    ctx.fill();
    ctx.restore();
  }
}

function KnowledgeGraphInner({
  data,
  width,
  height,
  selectedId,
  onSelectNode,
  queryUsedIds,
  queryUpdatedId,
  reheatToken,
  onLayoutSnapshot,
  birthEpoch = 0,
  birthNodeIds = NO_BIRTH,
  queryPulseEpoch = 0,
  queryPulseIds = NO_BIRTH,
  fitAfterQueryEpoch = 0,
  focusCameraRequest = null,
  structuralUpdatePulse = 0,
  pathNodeIds = NO_PATH_IDS,
  pathLinkKeys = NO_PATH_IDS,
}: Props) {
  const fgRef = useRef<ForceGraphMethods<GraphNode, GraphLink> | undefined>(undefined);
  const wrapRef = useRef<HTMLDivElement>(null);
  const dataRef = useRef(data);
  dataRef.current = data;
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const fittedRef = useRef(false);
  const birthStartsRef = useRef<Map<string, number>>(new Map());
  const queryPulseStartsRef = useRef<Map<string, number>>(new Map());
  const [birthRafActive, setBirthRafActive] = useState(false);
  const [queryPulseRafActive, setQueryPulseRafActive] = useState(false);
  const hoverDimRef = useRef(0);

  const interactionFocusId = hoverId ?? selectedId;
  const hasQueryHighlight = queryUsedIds.size > 0 || Boolean(queryUpdatedId);
  const hasPathHighlight = pathNodeIds.size > 0;
  /** Focus mode (fade + link filter) applies only after a node is selected, not on mere hover. */
  const dimmingFocusId = hasQueryHighlight || hasPathHighlight ? null : selectedId;

  const { focusNeighborNodes, focusLinkKeys } = useMemo(() => {
    const { nodes, linkKeys } = neighborSets(data, dimmingFocusId);
    return { focusNeighborNodes: nodes, focusLinkKeys: linkKeys };
  }, [data, dimmingFocusId]);

  const restrictLinksToFocus = Boolean(dimmingFocusId) && !hasQueryHighlight;

  const graphAccessorRef = useRef<GraphAccessorCtx | undefined>(undefined);
  graphAccessorRef.current = {
    data,
    dimmingFocusId,
    focusNeighborNodes,
    focusLinkKeys,
    hoverDimRef,
    restrictLinksToFocus,
    pathNodeIds,
    pathLinkKeys,
  };

  const hoverDimRafRef = useRef(0);
  useEffect(() => {
    const target = dimmingFocusId ? 1 : 0;
    let stopped = false;

    const tick = () => {
      if (stopped) return;
      const cur = hoverDimRef.current;
      const next = cur + (target - cur) * 0.17;
      const arrived = Math.abs(target - next) < 0.0025;
      hoverDimRef.current = arrived ? target : next;
      fgRef.current?.resumeAnimation();
      if (!arrived) hoverDimRafRef.current = requestAnimationFrame(tick);
    };
    hoverDimRafRef.current = requestAnimationFrame(tick);
    return () => {
      stopped = true;
      cancelAnimationFrame(hoverDimRafRef.current);
    };
  }, [dimmingFocusId]);

  useEffect(() => {
    if (!structuralUpdatePulse) return;
    const el = wrapRef.current;
    if (!el) return;
    el.classList.remove("graph-wrap--structural-pulse");
    void el.offsetWidth;
    el.classList.add("graph-wrap--structural-pulse");
    const tid = window.setTimeout(() => {
      el.classList.remove("graph-wrap--structural-pulse");
    }, 900);
    return () => window.clearTimeout(tid);
  }, [structuralUpdatePulse]);

  useEffect(() => {
    if (!fitAfterQueryEpoch) return;
    const fg = fgRef.current;
    if (!fg || data.nodes.length === 0) return;
    const tid = window.setTimeout(() => {
      fg.zoomToFit(900, 72);
      fg.resumeAnimation();
    }, 120);
    return () => window.clearTimeout(tid);
  }, [fitAfterQueryEpoch, data.nodes.length]);

  const runFocusCamera = useCallback((node: GraphNode) => {
    const fg = fgRef.current;
    if (!fg || node.x === undefined || node.y === undefined) return;
    fg.centerAt(node.x, node.y, 1000);
    const cur = typeof fg.zoom === "function" ? fg.zoom() : 1;
    const target = Math.min(1.62, Math.max(1.45, typeof cur === "number" ? Math.max(cur * 1.14, 1.45) : 1.52));
    fg.zoom(target, 1000);
    fg.resumeAnimation();
  }, []);

  useEffect(() => {
    if (focusCameraRequest == null) return;
    const { nodeId, nonce } = focusCameraRequest;
    if (!nodeId || nonce < 1) return;
    const t = window.setTimeout(() => {
      const n = dataRef.current.nodes.find((x) => x.id === nodeId);
      if (!n) return;
      runFocusCamera(n);
    }, 60);
    return () => window.clearTimeout(t);
  }, [focusCameraRequest, runFocusCamera]);

  const paintCtx: PaintCtx = useMemo(
    () => ({
      interactionFocusId,
      dimmingFocusId,
      selectedId,
      queryUsedIds,
      queryUpdatedId,
      data,
      birthStartsRef,
      queryPulseStartsRef,
      focusNeighborNodes,
      hoverDimRef,
    }),
    [data, dimmingFocusId, focusNeighborNodes, interactionFocusId, queryUsedIds, queryUpdatedId, selectedId],
  );

  useEffect(() => {
    fittedRef.current = false;
  }, [data.nodes.length]);

  useEffect(() => {
    if (!birthNodeIds.length || birthEpoch === 0) return;

    const now = performance.now();
    birthStartsRef.current.clear();
    for (const id of birthNodeIds) {
      if (dataRef.current.nodes.some((n) => n.id === id)) {
        birthStartsRef.current.set(id, now);
      }
    }
    if (birthStartsRef.current.size === 0) return;

    setBirthRafActive(true);
    const tid = window.setTimeout(() => {
      birthStartsRef.current.clear();
      setBirthRafActive(false);
    }, BIRTH_TOTAL_MS + 100);

    return () => {
      window.clearTimeout(tid);
      birthStartsRef.current.clear();
      setBirthRafActive(false);
    };
  }, [birthEpoch, birthNodeIds.join("|")]);

  useEffect(() => {
    if (!queryPulseEpoch) return;

    const ids = [...new Set(queryPulseIds.map((s) => s.trim()).filter(Boolean))].filter((id) =>
      dataRef.current.nodes.some((n) => n.id === id),
    );
    if (!ids.length) return;

    const now = performance.now();
    queryPulseStartsRef.current.clear();
    for (const id of ids) {
      queryPulseStartsRef.current.set(id, now);
    }

    setQueryPulseRafActive(true);
    const tid = window.setTimeout(() => {
      queryPulseStartsRef.current.clear();
      setQueryPulseRafActive(false);
    }, QUERY_PULSE_TOTAL_MS + 80);

    return () => {
      window.clearTimeout(tid);
      queryPulseStartsRef.current.clear();
      setQueryPulseRafActive(false);
    };
  }, [queryPulseEpoch, queryPulseIds]);

  useLayoutEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    const charge = fg.d3Force("charge") as { strength?: (s?: number) => void } | undefined;
    charge?.strength?.(-120);

    const link = fg.d3Force("link") as { distance?: (d: number) => void } | undefined;
    link?.distance?.(90);

    fg.d3Force("center", forceCenter(0, 0).strength(0.055));

    fg.d3Force("collide", forceCollide<GraphNode>().radius(28).strength(0.72));
  }, [width, height, data.nodes.length, data.links.length, reheatToken]);

  useEffect(() => {
    const t = window.setTimeout(() => {
      fgRef.current?.d3ReheatSimulation?.();
    }, 48);
    return () => window.clearTimeout(t);
  }, [reheatToken]);

  const nodeVal = useCallback(
    (n: GraphNode) => {
      const deg = n.__deg ?? 0;
      let v = 0.92 + Math.sqrt(deg + 1) * 0.34;
      if (deg >= 6) v += 0.12;
      if (hoverId === n.id) v *= 1.1;
      if (selectedId === n.id) v *= 1.14;
      if (queryUpdatedId && queryUpdatedId === n.id) v *= 1.05;
      if (queryUsedIds.has(n.id)) v *= 1.04;

      const start = birthStartsRef.current.get(n.id);
      if (start !== undefined) {
        const { scale } = birthVisual(performance.now() - start);
        v *= scale;
      }
      const pStart = queryPulseStartsRef.current.get(n.id);
      if (pStart !== undefined) {
        v *= queryPulseScale(performance.now() - pStart);
      }
      return v;
    },
    [hoverId, queryUsedIds, queryUpdatedId, selectedId],
  );

  const nodeColor = useCallback((n: GraphNode) => {
    const g = graphAccessorRef.current;
    if (!g) return fadeRgbColor(cssColorToRgbString(groupColor(n.group, false)), 0.88);
    const id = n.id;
    if (g.pathNodeIds.size > 0) {
      if (g.pathNodeIds.has(id)) return fadeRgbColor(cssColorToRgbString(PATH_HIGHLIGHT_COLOR), 1);
      return fadeRgbColor(cssColorToRgbString(groupColor(n.group, false)), 0.2);
    }
    const dim = g.hoverDimRef.current;
    const baseCss = n.color ?? groupColor(n.group, false);
    const rgb = cssColorToRgbString(baseCss);
    if (!g.dimmingFocusId || g.focusNeighborNodes.has(id)) return fadeRgbColor(rgb, 0.92);
    const floor = g.restrictLinksToFocus ? 0.16 : 0.2;
    const opacityMult = Math.max(floor, 1 - dim * 0.88);
    return fadeRgbColor(rgb, opacityMult);
  }, []);

  const linkColor = useCallback((link: GraphLink) => {
    const g = graphAccessorRef.current;
    const s = typeof link.source === "string" ? link.source : link.source.id;
    const t = typeof link.target === "string" ? link.target : link.target.id;
    const k = linkKey(s, t);
    if (!g) return "rgba(120, 135, 160, 0.26)";
    if (g.pathLinkKeys.size > 0) {
      return g.pathLinkKeys.has(k) ? "rgba(79, 209, 255, 0.92)" : "rgba(120, 135, 160, 0.08)";
    }
    const dim = g.hoverDimRef.current;
    const inN = g.focusLinkKeys.has(k);
    const fadeOut = Math.max(0.06, 1 - dim * 0.9);
    if (inN) {
      const a = 0.4 + dim * 0.34;
      return `rgba(186, 198, 255, ${a})`;
    }
    return `rgba(120, 135, 160, ${0.28 * fadeOut})`;
  }, []);

  const linkWidth = useCallback((link: GraphLink) => {
    const g = graphAccessorRef.current;
    const s = typeof link.source === "string" ? link.source : link.source.id;
    const t = typeof link.target === "string" ? link.target : link.target.id;
    const k = linkKey(s, t);
    if (!g) return 0.26;
    if (g.pathLinkKeys.size > 0) {
      return g.pathLinkKeys.has(k) ? 2.4 : 0.12;
    }
    const dim = g.hoverDimRef.current;
    const inN = g.focusLinkKeys.has(k);
    if (inN) return 0.58 + dim * 0.44;
    return Math.max(0.1, 0.28 - dim * 0.14);
  }, []);

  const linkVisibility = useCallback((link: GraphLink) => {
    const g = graphAccessorRef.current;
    if (!g?.restrictLinksToFocus) return true;
    const s = typeof link.source === "string" ? link.source : link.source.id;
    const t = typeof link.target === "string" ? link.target : link.target.id;
    return g.focusLinkKeys.has(linkKey(s, t));
  }, []);

  const nodeLabel = useCallback(
    (n: GraphNode) => {
      const deg = n.__deg ?? 0;
      if (hoverId === n.id) return n.title ?? n.id;
      if (deg >= 12) return n.title ?? n.id;
      return "";
    },
    [hoverId],
  );

  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      onSelectNode(node);
      runFocusCamera(node);
    },
    [onSelectNode, runFocusCamera],
  );

  const hoverNode = hoverId ? data.nodes.find((n) => n.id === hoverId) : undefined;

  return (
    <div
      className="graph-wrap graph-wrap--animated"
      ref={wrapRef}
      onMouseMove={(e) => {
        if (!hoverId || !wrapRef.current) return;
        const r = wrapRef.current.getBoundingClientRect();
        setTooltipPos({ x: e.clientX - r.left, y: e.clientY - r.top });
      }}
      onMouseLeave={() => setHoverId(null)}
    >
      <ForceGraph2D
        ref={fgRef}
        width={width}
        height={height}
        graphData={data}
        backgroundColor="rgba(0,0,0,0)"
        nodeId="id"
        nodeAutoColorBy="group"
        nodeColor={nodeColor}
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkVisibility={linkVisibility}
        linkDirectionalArrowLength={4.5}
        linkDirectionalArrowRelPos={1}
        linkDirectionalArrowColor={linkColor}
        nodeLabel={nodeLabel}
        nodeVal={nodeVal}
        nodeRelSize={3.15}
        cooldownTicks={300}
        cooldownTime={4000}
        d3AlphaMin={0.01}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.4}
        minZoom={0.5}
        maxZoom={2}
        autoPauseRedraw={!birthRafActive && !queryPulseRafActive}
        onNodeHover={(node) => {
          setHoverId(node?.id ?? null);
        }}
        onNodeClick={(node) => {
          handleNodeClick(node as GraphNode);
        }}
        onBackgroundClick={() => onSelectNode(null)}
        onEngineStop={() => {
          onLayoutSnapshot?.();
          const fg = fgRef.current;
          if (fg && !fittedRef.current && data.nodes.length > 0) {
            fittedRef.current = true;
            fg.zoomToFit(900, 72);
          }
        }}
        nodeCanvasObjectMode={(node) => nodeCanvasMode(node as GraphNode, paintCtx)}
        nodeCanvasObject={(node, ctx, globalScale) => {
          paintNodeCanvas(node as GraphNode, ctx, globalScale, paintCtx);
        }}
      />
      {hoverNode ? (
        <div
          className="graph-tooltip"
          style={{ left: tooltipPos.x + 12, top: tooltipPos.y + 12 }}
          role="status"
        >
          <div className="graph-tooltip__title">{hoverNode.title ?? hoverNode.id}</div>
          {hoverNode.summary ? <div className="graph-tooltip__summary">{hoverNode.summary}</div> : null}
        </div>
      ) : null}
    </div>
  );
}

export const KnowledgeGraph = memo(KnowledgeGraphInner);
