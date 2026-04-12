import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import type { GraphNode } from "../types";

type Props = {
  nodes: GraphNode[];
  onNavigateToNode: (node: GraphNode) => void;
};

export function GraphSearchBar({ nodes, onNavigateToNode }: Props) {
  const id = useId();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase();
    if (!t) return [];
    return nodes
      .filter((n) => {
        const title = (n.title ?? n.id).toLowerCase();
        const idLower = n.id.toLowerCase();
        return title.includes(t) || idLower.includes(t);
      })
      .slice(0, 8);
  }, [nodes, q]);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const pick = useCallback(
    (n: GraphNode) => {
      onNavigateToNode(n);
      setQ("");
      setOpen(false);
    },
    [onNavigateToNode],
  );

  return (
    <div className="graph-search" ref={wrapRef}>
      <label htmlFor={`${id}-search`} className="visually-hidden">
        Jump to concept
      </label>
      <input
        id={`${id}-search`}
        type="search"
        className="graph-search__input"
        placeholder="Jump to concept…"
        value={q}
        autoComplete="off"
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
      />
      {open && filtered.length > 0 ? (
        <ul className="graph-search__results" role="listbox">
          {filtered.map((n) => (
            <li key={n.id} role="option">
              <button type="button" className="graph-search__pick" onMouseDown={(e) => e.preventDefault()} onClick={() => pick(n)}>
                <span className="graph-search__pick-title">{n.title ?? n.id}</span>
                {n.summary ? <span className="graph-search__pick-sub">{n.summary.slice(0, 72)}{n.summary.length > 72 ? "…" : ""}</span> : null}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
