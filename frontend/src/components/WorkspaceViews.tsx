import { useState } from "react";
import type { GraphNode, Proposal, StatsResponse, TopicStatus, Zone } from "../types";
import { ProposalDetails } from "./ProposalDetails";

const STATUS_LABEL: Record<TopicStatus, string> = {
  not_started: "Not started",
  in_progress: "In progress",
  complete: "Complete",
};

type HomeProps = {
  stats: StatsResponse | null;
  statsLoading: boolean;
  graphLoading: boolean;
  graphError: string | null;
  graphNodes: GraphNode[];
  pendingProposals: Proposal[];
  proposalsLoading: boolean;
  proposalsError: string | null;
  zones: Zone[];
  onOpenAiOperations: () => void;
  onOpenLearn: () => void;
  onOpenExplore: () => void;
  onOpenReview: () => void;
  onOpenTopicInLearn: (id: string) => void;
  onRetryGraph: () => void;
};

export function HomeWorkspace({
  stats,
  statsLoading,
  graphLoading,
  graphError,
  graphNodes,
  pendingProposals,
  proposalsLoading,
  proposalsError,
  zones,
  onOpenAiOperations,
  onOpenLearn,
  onOpenExplore,
  onOpenReview,
  onOpenTopicInLearn,
  onRetryGraph,
}: HomeProps) {
  const inProgressTopics = graphNodes
    .filter((node) => node.status === "in_progress")
    .sort((a, b) => (a.title ?? a.id).localeCompare(b.title ?? b.id));
  const hasTopics = graphNodes.length > 0;
  const showEmptyState = !graphLoading && !graphError && !hasTopics;
  const showGraphError = !graphLoading && Boolean(graphError) && !hasTopics;
  const showGraphRefreshWarning = !graphLoading && Boolean(graphError) && hasTopics;
  const topicCount = statsLoading ? "—" : stats?.total_nodes ?? "—";
  const dependencyCount = statsLoading ? "—" : stats?.total_edges ?? "—";
  const proposalCount = proposalsLoading || proposalsError ? "—" : pendingProposals.length;

  return (
    <section className="workspace-view workspace-view--home" aria-labelledby="home-title">
      <header className="workspace-view__header">
        <p className="workspace-view__eyebrow">Knowledge overview</p>
        <h1 id="home-title">Your knowledge graph</h1>
        <p>Continue learning, review proposed changes, or explore how your topics connect.</p>
      </header>

      <div className="home-overview__actions" aria-label="Knowledge actions">
        <button type="button" className="workspace-view__action" onClick={onOpenAiOperations}>
          + Add knowledge
        </button>
        <button type="button" className="workspace-view__action workspace-view__action--quiet" onClick={onOpenLearn}>
          Learn
        </button>
        <button type="button" className="workspace-view__action workspace-view__action--quiet" onClick={onOpenExplore}>
          Explore graph
        </button>
        {!proposalsLoading && pendingProposals.length > 0 ? (
          <button type="button" className="workspace-view__action workspace-view__action--quiet" onClick={onOpenReview}>
            Review ({pendingProposals.length})
          </button>
        ) : null}
      </div>

      <div className="workspace-view__summary" aria-label="Knowledge summary">
        <div>
          <span className="workspace-view__summary-value">{topicCount}</span>
          <span className="workspace-view__summary-label">Topics</span>
        </div>
        <div>
          <span className="workspace-view__summary-value">{dependencyCount}</span>
          <span className="workspace-view__summary-label">Dependencies</span>
        </div>
        {zones.length > 0 ? (
          <div>
            <span className="workspace-view__summary-value">{zones.length}</span>
            <span className="workspace-view__summary-label">Zones</span>
          </div>
        ) : null}
        <div>
          <span className="workspace-view__summary-value">{proposalCount}</span>
          <span className="workspace-view__summary-label">Pending review</span>
        </div>
      </div>

      {graphLoading && !hasTopics ? <p className="workspace-view__muted">Loading your knowledge graph…</p> : null}

      {showGraphRefreshWarning ? (
        <div className="home-overview__warning" role="alert">
          <p>
            The latest graph refresh failed. Showing the last loaded topic information. {graphError}
          </p>
          <button type="button" className="workspace-view__action" onClick={onRetryGraph}>
            Retry
          </button>
        </div>
      ) : null}

      {showGraphError ? (
        <section className="workspace-view__section workspace-view__section--quiet" aria-labelledby="home-graph-error-title">
          <p className="workspace-view__eyebrow">Graph unavailable</p>
          <h2 id="home-graph-error-title">Your knowledge could not be loaded</h2>
          <p className="workspace-view__muted">{graphError}</p>
        </section>
      ) : null}

      {showEmptyState ? (
        <section className="workspace-view__section workspace-view__section--quiet" aria-labelledby="home-empty-title">
          <p className="workspace-view__eyebrow">A clear starting point</p>
          <h2 id="home-empty-title">Add knowledge to begin</h2>
          <p className="workspace-view__muted">
            Synapse turns your notes and ideas into a directed knowledge graph you can study and explore.
          </p>
          <div className="home-overview__actions">
            <button type="button" className="workspace-view__action" onClick={onOpenAiOperations}>
              + Add knowledge
            </button>
            <button type="button" className="workspace-view__action workspace-view__action--quiet" onClick={onOpenExplore}>
              Explore
            </button>
          </div>
        </section>
      ) : null}

      {hasTopics ? (
        <section className="workspace-view__section" aria-labelledby="home-continue-title">
        <div className="workspace-view__section-heading">
          <h2 id="home-continue-title">Continue learning</h2>
          <span>{inProgressTopics.length ? `${inProgressTopics.length} in progress` : "No active topics"}</span>
        </div>
        {inProgressTopics.length === 0 ? (
          <p className="workspace-view__muted">
            No topics are currently in progress. Open Learn to choose a topic and begin.
          </p>
        ) : (
          <div className="workspace-topic-list">
            {inProgressTopics.map((topic) => (
              <button
                type="button"
                className="workspace-topic-list__item"
                key={topic.id}
                onClick={() => onOpenTopicInLearn(topic.id)}
              >
                <span>{topic.title ?? topic.id}</span>
                <small>{STATUS_LABEL[topic.status ?? "not_started"]}</small>
              </button>
            ))}
          </div>
        )}
        </section>
      ) : null}

      {pendingProposals.length > 0 || proposalsError ? (
        <section className="workspace-view__section workspace-view__section--quiet" aria-labelledby="home-review-title">
          <div className="workspace-view__section-heading">
            <h2 id="home-review-title">Pending review</h2>
            <span>{proposalsError ? "Unavailable" : `${pendingProposals.length} waiting`}</span>
          </div>
          {proposalsError ? (
            <p className="workspace-view__muted">Pending proposals could not be loaded. Open Review to try again.</p>
          ) : (
            <>
              <ul className="home-overview__proposal-preview">
                {pendingProposals.slice(0, 3).map((proposal) => (
                  <li key={proposal.id} className="home-overview__proposal-item">
                    <span>{proposal.source || "Untitled proposal"}</span>
                    <small>{proposal.mode}</small>
                  </li>
                ))}
              </ul>
              {pendingProposals.length > 3 ? (
                <p className="workspace-view__muted">and {pendingProposals.length - 3} more waiting in Review.</p>
              ) : null}
            </>
          )}
          <button type="button" className="workspace-view__action workspace-view__action--quiet" onClick={onOpenReview}>
            Open Review
          </button>
        </section>
      ) : null}
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
