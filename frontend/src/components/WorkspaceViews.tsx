import { useState } from "react";
import type { GraphNode, Proposal, StatsResponse, TopicStatus } from "../types";
import { ProposalDetails } from "./ProposalDetails";

const STATUS_LABEL: Record<TopicStatus, string> = {
  not_started: "Not started",
  in_progress: "In progress",
  complete: "Complete",
};

type HomeProps = {
  stats: StatsResponse | null;
  loading: boolean;
  graphNodes: GraphNode[];
  onOpenTopic: (id: string) => void;
};

export function HomeWorkspace({ stats, loading, graphNodes, onOpenTopic }: HomeProps) {
  const isTopicAvailable = (id: string) => graphNodes.some((node) => node.id === id);

  return (
    <section className="workspace-view workspace-view--home" aria-labelledby="home-title">
      <header className="workspace-view__header">
        <p className="workspace-view__eyebrow">Workspace</p>
        <h1 id="home-title">What would you like to learn?</h1>
        <p>Continue with a topic or open Explore to see how your knowledge connects.</p>
      </header>

      <div className="workspace-view__summary" aria-label="Knowledge summary">
        <div>
          <span className="workspace-view__summary-value">{loading ? "—" : stats?.total_nodes ?? 0}</span>
          <span className="workspace-view__summary-label">Topics</span>
        </div>
        <div>
          <span className="workspace-view__summary-value">{loading ? "—" : stats?.total_edges ?? 0}</span>
          <span className="workspace-view__summary-label">Dependencies</span>
        </div>
      </div>

      <section className="workspace-view__section" aria-labelledby="home-recent-title">
        <div className="workspace-view__section-heading">
          <h2 id="home-recent-title">Recent topics</h2>
          <span>{loading ? "Loading…" : `${stats?.recent_nodes.length ?? 0} available`}</span>
        </div>
        {!stats?.recent_nodes.length ? (
          <p className="workspace-view__muted">
            {loading ? "Loading your topics…" : "No topics yet. Use Add knowledge to create your first learning graph."}
          </p>
        ) : (
          <div className="workspace-topic-list">
            {stats.recent_nodes.map((topic) => (
              <button
                type="button"
                className="workspace-topic-list__item"
                key={topic.id}
                onClick={() => onOpenTopic(topic.id)}
                disabled={!isTopicAvailable(topic.id)}
                title={isTopicAvailable(topic.id) ? undefined : "This topic will be available when the graph is ready."}
              >
                <span>{topic.title}</span>
                <small>{STATUS_LABEL[topic.status]}</small>
              </button>
            ))}
          </div>
        )}
      </section>
    </section>
  );
}

type LearnProps = {
  nodes: GraphNode[];
  onOpenTopic: (id: string) => void;
};

export function LearnWorkspace({ nodes, onOpenTopic }: LearnProps) {
  const sortedNodes = [...nodes].sort((a, b) => (a.title ?? a.id).localeCompare(b.title ?? b.id));

  return (
    <section className="workspace-view" aria-labelledby="learn-title">
      <header className="workspace-view__header">
        <p className="workspace-view__eyebrow">Learn</p>
        <h1 id="learn-title">Your topics</h1>
        <p>Choose a topic to open its summary, learning context, resources, notes, questions, and quiz.</p>
      </header>

      {!sortedNodes.length ? (
        <p className="workspace-view__muted">No topics yet. Add knowledge to begin.</p>
      ) : (
        <div className="workspace-topic-list workspace-topic-list--grid">
          {sortedNodes.map((node) => (
            <button type="button" className="workspace-topic-list__item" key={node.id} onClick={() => onOpenTopic(node.id)}>
              <span>{node.title ?? node.id}</span>
              <small>{node.status ? STATUS_LABEL[node.status] : "Not started"}</small>
              {node.summary ? <em>{node.summary}</em> : null}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

type ReviewProps = {
  pendingProposals: Proposal[];
  loading: boolean;
  error: string | null;
  nodes: GraphNode[];
  onOpenAiOperations: () => void;
  onRefresh: () => void;
  onApplyProposal: (proposalId: string) => Promise<void>;
  onDiscardProposal: (proposalId: string) => Promise<void>;
};

export function ReviewWorkspace({
  pendingProposals,
  loading,
  error,
  nodes,
  onOpenAiOperations,
  onRefresh,
  onApplyProposal,
  onDiscardProposal,
}: ReviewProps) {
  const [busyProposalId, setBusyProposalId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionErrorProposalId, setActionErrorProposalId] = useState<string | null>(null);

  async function runAction(proposalId: string, action: (id: string) => Promise<void>) {
    if (busyProposalId) return;
    setBusyProposalId(proposalId);
    setActionError(null);
    setActionErrorProposalId(null);
    try {
      await action(proposalId);
    } catch (actionFailure) {
      setActionError(actionFailure instanceof Error ? actionFailure.message : "The proposal could not be updated");
      setActionErrorProposalId(proposalId);
    } finally {
      setBusyProposalId(null);
    }
  }

  return (
    <section className="workspace-view workspace-view--review" aria-labelledby="review-title">
      <header className="workspace-view__header">
        <p className="workspace-view__eyebrow">Review</p>
        <h1 id="review-title">Review your knowledge</h1>
        <p>Read proposed knowledge changes before they become part of your graph.</p>
      </header>

      <div className="review-workspace__toolbar">
        <div>
          <h2>Pending proposals</h2>
          <p>Apply changes you trust, or discard them without changing the graph.</p>
        </div>
        <div className="review-workspace__toolbar-actions">
          <button type="button" className="workspace-view__action" onClick={onOpenAiOperations}>
            Add knowledge
          </button>
          <button type="button" className="workspace-view__action workspace-view__action--quiet" onClick={onRefresh} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {error ? (
        <div className="review-workspace__error" role="alert">
          <p>Could not load pending proposals: {error}</p>
          <button type="button" className="workspace-view__action" onClick={onRefresh} disabled={loading}>
            Try again
          </button>
        </div>
      ) : null}

      {actionError ? <p className="review-workspace__action-error" role="alert">{actionError}</p> : null}

      {loading && pendingProposals.length === 0 ? (
        <p className="workspace-view__muted">Loading pending proposals…</p>
      ) : null}

      {!loading && !error && pendingProposals.length === 0 ? (
        <section className="review-workspace__empty" aria-live="polite">
          <p className="review-workspace__empty-kicker">Nothing waiting</p>
          <h2>No pending proposals</h2>
          <p>Generate knowledge with Add knowledge and its proposal will appear here for review.</p>
        </section>
      ) : null}

      {pendingProposals.length > 0 ? (
        <div className="review-workspace__queue">
          {pendingProposals.map((proposal) => (
            <article className="review-proposal" key={proposal.id}>
              <header className="review-proposal__header">
                <div>
                  <p className="review-proposal__kicker">Pending proposal</p>
                  <h2>{proposal.source}</h2>
                </div>
                <span className="review-proposal__status">Pending</span>
              </header>
              <p className="review-proposal__meta">
                {proposal.mode} · {proposal.created_at ? new Date(proposal.created_at).toLocaleString() : "Saved proposal"}
              </p>
              <ProposalDetails
                proposal={proposal}
                nodes={nodes}
                busy={busyProposalId === proposal.id}
                error={actionErrorProposalId === proposal.id ? actionError : null}
                onApply={() => void runAction(proposal.id, onApplyProposal)}
                onDiscard={() => void runAction(proposal.id, onDiscardProposal)}
              />
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}
