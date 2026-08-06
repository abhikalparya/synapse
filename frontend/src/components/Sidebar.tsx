import type { RecentNode, StatsResponse } from "../types";

type Props = {
  stats: StatsResponse | null;
  loading: boolean;
  onPickNode: (id: string) => void;
};

const STATUS_LABEL: Record<RecentNode["status"], string> = {
  not_started: "Not started",
  in_progress: "In progress",
  complete: "Complete",
};

export function Sidebar({ stats, loading, onPickNode }: Props) {
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
