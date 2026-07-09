from pydantic import BaseModel


class TicketRequest(BaseModel):
    ticket_text: str
    thread_id: str
    conversation_history: list[dict] = []


class ClassificationOut(BaseModel):
    category: str
    severity: str
    routing_path: str
    confidence: float
    reasoning: str


class TicketResponse(BaseModel):
    final_response: str
    classification: ClassificationOut
    conversation_history: list[dict]
    trace_id: str
    submitted_at: str
    total_cost_usd: float  # SLO #14 — cost per ticket (avg <= $0.05, cap $0.15)
