import type { RecentNode, RefactorResponse, StatsResponse } from "../types";

type Props = {
  stats: StatsResponse | null;
  loading: boolean;
  onPickTitle: (title: string) => void;
  onAddNote: () => void;
  onRefactor: () => void;
  refactorLoading: boolean;
  refactorError: string | null;
  refactorResult: RefactorResponse | null;
};

export function Sidebar({
  stats,
  loading,
  onPickTitle,
  onAddNote,
  onRefactor,
  refactorLoading,
  refactorError,
  refactorResult,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="sidebar__logo" aria-hidden>
          ◈
        </span>
        <div>
          <div className="sidebar__name">Synapse</div>
          <div className="sidebar__tag">Knowledge graph</div>
        </div>
      </div>

      <button type="button" className="sidebar__add-note" onClick={onAddNote}>
        + Add Knowledge
      </button>

      <div className="sidebar__refactor-block">
        <button
          type="button"
          className="sidebar__refactor"
          onClick={onRefactor}
          disabled={refactorLoading}
          aria-busy={refactorLoading}
        >
          {refactorLoading ? (
            <>
              <span className="sidebar__refactor-spinner" aria-hidden />
              Refactoring knowledge…
            </>
          ) : (
            <>
              <span className="sidebar__refactor-icon" aria-hidden>
                <svg
                  className="sidebar__refactor-icon-svg"
                  viewBox="0 0 24 24"
                  width="18"
                  height="18"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    d="M7.6 16.7 Q9.9 12.6 12 9.35 Q14.1 12.6 16.4 16.7"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <circle cx="12" cy="8.65" r="2.4" stroke="currentColor" strokeWidth="1.5" />
                  <circle cx="7.45" cy="16.85" r="2.1" stroke="currentColor" strokeWidth="1.4" />
                  <circle cx="16.55" cy="16.85" r="2.1" stroke="currentColor" strokeWidth="1.4" />
                  <path
                    d="M12 3.85v1.35M11.05 4.52h1.9"
                    stroke="currentColor"
                    strokeWidth="1.2"
                    strokeLinecap="round"
                    opacity="0.95"
                  />
                </svg>
              </span>
              Refactor Knowledge
            </>
          )}
        </button>
        {refactorError ? <p className="sidebar__refactor-error">{refactorError}</p> : null}
        {refactorResult && !refactorLoading ? (
          <div className="sidebar__refactor-summary">
            <div className="sidebar__refactor-summary-title">Refactor complete:</div>
            <ul>
              <li>{refactorResult.pages_rewritten} pages rewritten</li>
              <li>{refactorResult.pages_merged} merged</li>
              <li>{refactorResult.pages_updated} updated</li>
            </ul>
            {refactorResult.errors.length ? (
              <p className="sidebar__refactor-warn">{refactorResult.errors.slice(0, 2).join(" · ")}</p>
            ) : null}
          </div>
        ) : null}
      </div>

      <section className="sidebar__section">
        <h3>Graph stats</h3>
        {loading && !stats ? (
          <div className="skeleton skeleton--stats" />
        ) : stats ? (
          <div className="stat-grid">
            <div className="stat-tile">
              <span className="stat-tile__value">{stats.total_nodes}</span>
              <span className="stat-tile__label">Nodes</span>
            </div>
            <div className="stat-tile">
              <span className="stat-tile__value">{stats.total_edges}</span>
              <span className="stat-tile__label">Edges</span>
            </div>
          </div>
        ) : (
          <p className="sidebar__muted">No stats yet.</p>
        )}
      </section>

      <section className="sidebar__section sidebar__section--grow">
        <h3>Recent nodes</h3>
        {!stats?.recent_nodes.length ? (
          <p className="sidebar__muted">{loading ? "Loading…" : "No pages yet."}</p>
        ) : (
          <ul className="recent-list">
            {stats.recent_nodes.map((n: RecentNode) => (
              <li key={n.filename}>
                <button type="button" className="recent-list__btn" onClick={() => onPickTitle(n.title)}>
                  <span className="recent-list__title">{n.title}</span>
                  {n.tags.length ? (
                    <span className="recent-list__tags">{n.tags.slice(0, 2).join(" · ")}</span>
                  ) : null}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
