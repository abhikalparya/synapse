"""Prompt for the in-session Q&A assistant (Phase 10) -- answers a free-text question
scoped to a single topic's own summary/resources/artifacts. This is strictly read/explain
-only: never asked to propose or imply a graph mutation, unlike the ingest/expand/audit
/reshape modes."""

ASK_SYSTEM_PREAMBLE = """You are a study assistant helping a learner understand ONE topic \
in their knowledge graph. Answer the learner's question using only the topic context given \
below (its summary, attached resources, and anything they've previously produced while \
studying it).

Rules:
- Answer only the question asked, grounded in the context below. If the context doesn't \
cover something the question needs, say so plainly rather than inventing detail.
- You are explain-only: never propose adding, removing, or restructuring topics or \
dependencies, even if the learner asks you to. If they ask for a graph change, tell them \
to use the AI operations panel instead.
- Keep the answer focused and conversational -- plain text or light markdown, no JSON."""


def build_ask_prompt(
    *,
    topic_title: str,
    topic_summary: str,
    resources: list[dict[str, str]],
    artifacts: list[dict[str, str]],
    question: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    parts = [ASK_SYSTEM_PREAMBLE, f"Topic: {topic_title}\nSummary: {topic_summary or '(no summary)'}"]

    if resources:
        lines = "\n".join(f"- [{r['type']}] {r['title'] or r['source_ref']}" for r in resources)
        parts.append(f"Attached resources:\n{lines}")

    if artifacts:
        lines = "\n".join(
            f"- [{a['type']}] {a['title'] or '(untitled)'}: {a['content']}" for a in artifacts
        )
        parts.append(f"Things the learner has already produced for this topic:\n{lines}")

    if history:
        # Memory setting (Phase 13): prior turns in this topic's Q&A, most recent last.
        lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
        parts.append(f"Earlier questions you already answered in this conversation:\n{lines}")

    parts.append(f"Learner's question:\n{question.strip()}")
    return "\n\n".join(parts)
