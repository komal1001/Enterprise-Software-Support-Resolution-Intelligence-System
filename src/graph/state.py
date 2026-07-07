from typing import Optional, TypedDict


class TicketState(TypedDict):
    """
    Shared state that travels through every node in the LangGraph graph.
    Each agent reads from it and writes its result back into it.
    """
    ticket_text: str
    conversation_history: list[dict]     # Multi-turn chat — [{"role": "user"|"assistant", "content": "..."}]
    classification: Optional[dict]       # Agent 1 — category, severity, route, confidence, reasoning
    rag_result: Optional[dict]           # Agent 2 — retrieved docs + source attribution
    sql_result: Optional[dict]           # Agent 3 — account, ticket, incident data
    severity_assessment: Optional[dict]  # Agent 4 — re-evaluated severity + confidence
    escalation_package: Optional[dict]   # Agent 5 — full context package for human handoff
    final_response: Optional[str]        # Response Synthesizer — what the customer sees
    reflection_count: int                # tracks re-retrieval attempts — max 1 per ticket (prevents infinite loop)
