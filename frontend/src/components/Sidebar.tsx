import { useState, type FormEvent } from "react";
import type { WorkspaceView } from "./AppShell";
import type { RecentNode, StatsResponse, Zone } from "../types";

type Props = {
  activeView: WorkspaceView;
  onNavigate: (view: WorkspaceView) => void;
  stats: StatsResponse | null;
  loading: boolean;
  onPickNode: (id: string) => void;
  onOpenAiOperations: () => void;
  onOpenSettings: () => void;
  onUndoLastChange: () => void;
  undoBusy: boolean;
  zones: Zone[];
  onCreateZone: (label: string, color: string) => Promise<void>;
};

const STATUS_LABEL: Record<RecentNode["status"], string> = {
  not_started: "Not started",
  in_progress: "In progress",
  complete: "Complete",
};

const DEFAULT_ZONE_COLOR = "#8b5cf6";

export function Sidebar({
  activeView,
  onNavigate,
  stats,
  loading,
  onPickNode,
  onOpenAiOperations,
  onOpenSettings,
  onUndoLastChange,
  undoBusy,
  zones,
  onCreateZone,
}: Props) {
  const [zoneLabel, setZoneLabel] = useState("");
  const [zoneColor, setZoneColor] = useState(DEFAULT_ZONE_COLOR);
  const [zoneBusy, setZoneBusy] = useState(false);

  async function handleCreateZone(e: FormEvent) {
    e.preventDefault();
    const trimmed = zoneLabel.trim();
    if (!trimmed || zoneBusy) return;
    setZoneBusy(true);
    try {
      await onCreateZone(trimmed, zoneColor);
      setZoneLabel("");
    } finally {
      setZoneBusy(false);
    }
  }

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="sidebar__logo" aria-hidden>
          ◈
        </span>
        <div>
          <div className="sidebar__name">Synapse</div>
          <div className="sidebar__tag">Dependency graph</div>
        </div>
      </div>

      <nav className="sidebar__nav" aria-label="Primary navigation">
        {([
          ["home", "Home"],
          ["learn", "Learn"],
          ["explore", "Explore"],
          ["review", "Review"],
        ] as const).map(([view, label]) => (
          <button
            type="button"
            key={view}
            className={`sidebar__nav-item${activeView === view ? " sidebar__nav-item--active" : ""}`}
            onClick={() => onNavigate(view)}
            aria-current={activeView === view ? "page" : undefined}
          >
            {label}
          </button>
        ))}
      </nav>

      <button type="button" className="sidebar__add-note" onClick={onOpenAiOperations}>
        + Add knowledge
      </button>
      <button type="button" className="sidebar__undo" onClick={onUndoLastChange} disabled={undoBusy}>
        {undoBusy ? "Undoing…" : "Undo last change"}
      </button>
      <a className="sidebar__undo" href="/obsidian/export" download="synapse-export.zip">
        Export to Obsidian vault
      </a>
      <button type="button" className="sidebar__undo" onClick={onOpenSettings}>
        Settings
      </button>

      <section className="sidebar__section">
        <h3>Graph stats</h3>
        {loading && !stats ? (
          <div className="skeleton skeleton--stats" />
        ) : stats ? (
          <div className="stat-grid">
            <div className="stat-tile">
              <span className="stat-tile__value">{stats.total_nodes}</span>
              <span className="stat-tile__label">Topics</span>
            </div>
            <div className="stat-tile">
              <span className="stat-tile__value">{stats.total_edges}</span>
              <span className="stat-tile__label">Dependencies</span>
            </div>
          </div>
        ) : (
          <p className="sidebar__muted">No stats yet.</p>
        )}
      </section>

      <section className="sidebar__section">
        <h3>Zones</h3>
        {zones.length === 0 ? (
          <p className="sidebar__muted">No zones yet.</p>
        ) : (
          <ul className="zone-list">
            {zones.map((z) => (
              <li key={z.id} className="zone-list__item">
                <span className="zone-list__swatch" style={{ background: z.color ?? DEFAULT_ZONE_COLOR }} aria-hidden />
                {z.label}
              </li>
            ))}
          </ul>
        )}
        <form className="zone-form" onSubmit={handleCreateZone}>
          <input
            type="color"
            className="zone-form__color"
            value={zoneColor}
            onChange={(e) => setZoneColor(e.target.value)}
            disabled={zoneBusy}
            aria-label="Zone color"
          />
          <input
            type="text"
            className="zone-form__input"
            placeholder="New zone label"
            value={zoneLabel}
            onChange={(e) => setZoneLabel(e.target.value)}
            disabled={zoneBusy}
          />
          <button type="submit" className="zone-form__btn" disabled={zoneBusy || !zoneLabel.trim()}>
            +
          </button>
        </form>
      </section>

      <section className="sidebar__section sidebar__section--grow">
        <h3>Recent topics</h3>
        {!stats?.recent_nodes.length ? (
          <p className="sidebar__muted">{loading ? "Loading…" : "No topics yet."}</p>
        ) : (
          <ul className="recent-list">
            {stats.recent_nodes.map((n: RecentNode) => (
              <li key={n.id}>
                <button type="button" className="recent-list__btn" onClick={() => onPickNode(n.id)}>
                  <span className="recent-list__title">{n.title}</span>
                  <span className="recent-list__tags">{STATUS_LABEL[n.status]}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
