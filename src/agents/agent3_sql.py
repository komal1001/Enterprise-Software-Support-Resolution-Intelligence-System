from dotenv import load_dotenv

from src.graph.state import TicketState
from src.retrieval.sql import query

load_dotenv()


def agent3_sql(state: TicketState) -> TicketState:
    # sql.query() handles Langfuse logging internally (prompt version + span)
    state["sql_result"] = query(state["ticket_text"], state["classification"])
    return state
