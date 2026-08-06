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

/**
 * Greedy sparsify: keep edges with lower max-endpoint degree first until each node hits the cap.
 * Reduces dense tag-clique clutter while preserving weaker ties.
 */
export function capLinksPerNode(links: GraphLink[], maxPerNode: number): GraphLink[] {
  if (maxPerNode <= 0) return links;
  const deduped = dedupeLinks(links);
  const deg = new Map<string, number>();
  for (const l of deduped) {
    const s = linkEndpoint(l, "source");
    const t = linkEndpoint(l, "target");
    deg.set(s, (deg.get(s) ?? 0) + 1);
    deg.set(t, (deg.get(t) ?? 0) + 1);
  }
  type Scored = { s: string; t: string; score: number };
  const scored: Scored[] = deduped.map((l) => {
    const s = linkEndpoint(l, "source");
    const t = linkEndpoint(l, "target");
    return { s, t, score: Math.max(deg.get(s) ?? 0, deg.get(t) ?? 0) };
  });
  scored.sort((a, b) => a.score - b.score);
  const count = new Map<string, number>();
  const kept: GraphLink[] = [];
  for (const e of scored) {
    const cs = count.get(e.s) ?? 0;
    const ct = count.get(e.t) ?? 0;
    if (cs >= maxPerNode || ct >= maxPerNode) continue;
    kept.push({ source: e.s, target: e.t });
    count.set(e.s, cs + 1);
    count.set(e.t, ct + 1);
  }
  return kept;
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
  /** Max edges per node after dedupe (reduces tag-clique density) */
  maxLinksPerNode?: number;
};

export function prepareGraphData(raw: GraphData, options: PrepareGraphOptions = {}): GraphData {
  const maxLinks = options.maxLinksPerNode ?? 12;
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
  const capped = capLinksPerNode(normalizedLinks, maxLinks);
  return enrichGraphData({ nodes, links: capped });
}

/** Semantic hues by domain keyword (Obsidian-style); falls back to stable hash. */
export function groupColor(group: string, dim = false): string {
  const g = (group || "").toLowerCase();
  const s = dim ? 34 : 46;
  const l = dim ? 40 : 52;
  let hue = 255;
  if (/\b(ai|ml|llm|gpt|neural|model)\b/.test(g) || g.includes("artificial")) hue = 272;
  else if (/\b(eng|code|software|system|api|dev|tech)\b/.test(g) || g.includes("engineer")) hue = 218;
  else if (/\b(psych|mind|behavior|cognitive|mental)\b/.test(g)) hue = 152;
  else if (/\b(cross|meta|general|note)\b/.test(g) || g.includes("synapse")) hue = 48;
  else {
    let h = 0;
    for (let i = 0; i < group.length; i++) {
      h = (h * 31 + group.charCodeAt(i)) >>> 0;
    }
    hue = 220 + (h % 72);
  }
  return `hsl(${hue} ${s}% ${l}%)`;
}

/** Directly connected node ids (by wiki title id). */
export function neighborNodeIds(data: GraphData, nodeId: string): string[] {
  const out = new Set<string>();
  for (const l of data.links) {
    const s = linkEndpoint(l, "source");
    const t = linkEndpoint(l, "target");
    if (s === nodeId) out.add(t);
    if (t === nodeId) out.add(s);
  }
  return [...out];
}

export function neighborSets(
  data: GraphData,
  focusId: string | null,
): { nodes: Set<string>; linkKeys: Set<string> } {
  const nodes = new Set<string>();
  const linkKeys = new Set<string>();
  if (!focusId) return { nodes, linkKeys };
  nodes.add(focusId);
  for (const l of data.links) {
    const s = typeof l.source === "string" ? l.source : l.source.id;
    const t = typeof l.target === "string" ? l.target : l.target.id;
    if (s === focusId || t === focusId) {
      nodes.add(s);
      nodes.add(t);
      linkKeys.add(linkKey(s, t));
    }
  }
  return { nodes, linkKeys };
}

/** Visual radius for canvas overlays (matches compact “note” dots) */
export function nodeRadius(node: GraphNode, base = 3.1): number {
  const d = Math.max(0, node.__deg ?? 0);
  return base + Math.sqrt(d + 1) * 1.55;
}
