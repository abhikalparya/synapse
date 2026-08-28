import type { GraphData, GraphLink, GraphNode } from "./types";

export function linkEndpoint(link: GraphLink, end: "source" | "target"): string {
  const raw = link[end];
  if (typeof raw === "string") return raw;
  return raw.id;
}

/** Directed edge key: order matters (``source -> target``), unlike the old undirected graph. */
export function linkKey(source: string, target: string): string {
  return `${source}->${target}`;
}

export function dedupeLinks(links: GraphLink[]): GraphLink[] {
  const seen = new Set<string>();
  const out: GraphLink[] = [];
  for (const l of links) {
    const s = linkEndpoint(l, "source");
    const t = linkEndpoint(l, "target");
    if (s === t) continue;
    const k = linkKey(s, t);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push({ source: s, target: t });
  }
  return out;
}

export function enrichGraphData(raw: GraphData): GraphData {
  const deg = new Map<string, number>();
  for (const l of raw.links) {
    const s = linkEndpoint(l, "source");
    const t = linkEndpoint(l, "target");
    deg.set(s, (deg.get(s) ?? 0) + 1);
    deg.set(t, (deg.get(t) ?? 0) + 1);
  }
  const nodes = raw.nodes.map((n) => ({
    ...n,
    __deg: deg.get(n.id) ?? 0,
  }));
  const links = raw.links.map((l) => ({
    source: linkEndpoint(l, "source"),
    target: linkEndpoint(l, "target"),
  }));
  return { nodes, links };
}

export type PrepareGraphOptions = {
  /** Restore x,y after refetch to avoid a hard layout reset */
  snapshot?: Map<string, { x: number; y: number }>;
};

export function prepareGraphData(raw: GraphData, options: PrepareGraphOptions = {}): GraphData {
  const snap = options.snapshot;

  const nodes = raw.nodes.map((n) => {
    const base = { ...n };
    if (snap?.size) {
      const p = snap.get(n.id);
      if (p) {
        base.x = p.x;
        base.y = p.y;
        base.vx = 0;
        base.vy = 0;
      }
    }
    return base;
  });

  const normalizedLinks: GraphLink[] = raw.links.map((l) => ({
    source: linkEndpoint(l, "source"),
    target: linkEndpoint(l, "target"),
  }));
  return enrichGraphData({ nodes, links: dedupeLinks(normalizedLinks) });
}

export type DirectedRelationship = {
  source: string;
  target: string;
};

export type RelationshipSets = {
  prerequisiteIds: Set<string>;
  dependentIds: Set<string>;
  neighborIds: Set<string>;
  prerequisiteLinkKeys: Set<string>;
  dependentLinkKeys: Set<string>;
  linkKeys: Set<string>;
};

/**
 * Classify direct relationships using the graph's canonical direction:
 * source/dependent -> target/prerequisite.
 */
export function relationshipSets(
  relationships: readonly DirectedRelationship[],
  focusId: string | null,
): RelationshipSets {
  const prerequisiteIds = new Set<string>();
  const dependentIds = new Set<string>();
  const prerequisiteLinkKeys = new Set<string>();
  const dependentLinkKeys = new Set<string>();
  const linkKeys = new Set<string>();
  if (!focusId) {
    return {
      prerequisiteIds,
      dependentIds,
      neighborIds: new Set(),
      prerequisiteLinkKeys,
      dependentLinkKeys,
      linkKeys,
    };
  }

  for (const relationship of relationships) {
    const source = relationship.source;
    const target = relationship.target;
    if (source === target) continue;
    const key = linkKey(source, target);
    if (source === focusId) {
      prerequisiteIds.add(target);
      prerequisiteLinkKeys.add(key);
      linkKeys.add(key);
    } else if (target === focusId) {
      dependentIds.add(source);
      dependentLinkKeys.add(key);
      linkKeys.add(key);
    }
  }

  return {
    prerequisiteIds,
    dependentIds,
    neighborIds: new Set([focusId, ...prerequisiteIds, ...dependentIds]),
    prerequisiteLinkKeys,
    dependentLinkKeys,
    linkKeys,
  };
}

export function neighborSets(
  data: GraphData,
  focusId: string | null,
): RelationshipSets {
  return relationshipSets(
    data.links.map((link) => ({
      source: linkEndpoint(link, "source"),
      target: linkEndpoint(link, "target"),
    })),
    focusId,
  );
}

/** Visual radius for canvas overlays (matches compact “note” dots) */
export function nodeRadius(node: GraphNode, base = 3.1): number {
  const d = Math.max(0, node.__deg ?? 0);
  return base + Math.sqrt(d + 1) * 1.55;
}

export type Point = { x: number; y: number };

/** Andrew's monotone-chain convex hull, O(n log n). Returns points in CCW order. */
export function convexHull(points: Point[]): Point[] {
  if (points.length < 3) return points;
  const pts = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o: Point, a: Point, b: Point) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Point[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  upper.pop();
  lower.pop();
  return [...lower, ...upper];
}

/** Pushes each hull vertex outward from the centroid by `padding` px, so the drawn region
 * encloses its nodes with breathing room instead of connecting their exact centers. */
function padHullOutward(hull: Point[], padding: number): Point[] {
  const cx = hull.reduce((s, p) => s + p.x, 0) / hull.length;
  const cy = hull.reduce((s, p) => s + p.y, 0) / hull.length;
  return hull.map((p) => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    return { x: p.x + (dx / len) * padding, y: p.y + (dy / len) * padding };
  });
}

/** Traces a soft "blob" path through points via quadratic curves through edge
 * midpoints -- the standard trick for rounding a polygon without per-corner radius math. */
function tracePolygonSmooth(ctx: CanvasRenderingContext2D, points: Point[]) {
  const n = points.length;
  const mid = (a: Point, b: Point) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  const start = mid(points[n - 1], points[0]);
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  for (let i = 0; i < n; i++) {
    const p = points[i];
    const next = points[(i + 1) % n];
    const m = mid(p, next);
    ctx.quadraticCurveTo(p.x, p.y, m.x, m.y);
  }
  ctx.closePath();
}

/** Draws a soft grouping region behind a zone's member nodes -- a circle for one node, a
 * rounded capsule for two, a padded/smoothed convex hull for three or more. */
export function drawZoneRegion(
  ctx: CanvasRenderingContext2D,
  points: Point[],
  fillColor: string,
  strokeColor: string,
  padding = 34,
) {
  if (points.length === 0) return;

  if (points.length === 1) {
    ctx.beginPath();
    ctx.arc(points[0].x, points[0].y, padding, 0, Math.PI * 2);
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    return;
  }

  if (points.length === 2) {
    ctx.save();
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    ctx.lineTo(points[1].x, points[1].y);
    ctx.strokeStyle = fillColor;
    ctx.lineWidth = padding * 2;
    ctx.stroke();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
    return;
  }

  const padded = padHullOutward(convexHull(points), padding);
  tracePolygonSmooth(ctx, padded);
  ctx.fillStyle = fillColor;
  ctx.fill();
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}
