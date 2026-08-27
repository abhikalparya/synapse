import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";
import type { StatsResponse, Zone } from "../types";

export type WorkspaceView = "home" | "learn" | "explore" | "review";

type Props = {
  activeView: WorkspaceView;
  onNavigate: (view: WorkspaceView) => void;
  onPickNode: (id: string) => void;
  onOpenAiOperations: () => void;
  onOpenSettings: () => void;
  onUndoLastChange: () => void;
  undoBusy: boolean;
  stats: StatsResponse | null;
  statsLoading: boolean;
  zones: Zone[];
  onCreateZone: (label: string, color: string) => Promise<void>;
  children: ReactNode;
  contextPanel?: ReactNode;
};

export function AppShell({
  activeView,
  onNavigate,
  onPickNode,
  onOpenAiOperations,
  onOpenSettings,
  onUndoLastChange,
  undoBusy,
  stats,
  statsLoading,
  zones,
  onCreateZone,
  children,
  contextPanel,
}: Props) {
  return (
    <div className={`app${contextPanel ? " app--with-context" : ""}`}>
      <Sidebar
        activeView={activeView}
        onNavigate={onNavigate}
        stats={stats}
        loading={statsLoading}
        onPickNode={onPickNode}
        onOpenAiOperations={onOpenAiOperations}
        onOpenSettings={onOpenSettings}
        onUndoLastChange={onUndoLastChange}
        undoBusy={undoBusy}
        zones={zones}
        onCreateZone={onCreateZone}
      />
      {children}
      {contextPanel}
    </div>
  );
}
