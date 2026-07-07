import os
import threading
import time
from typing import Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool
import sqlglot
import sqlglot.expressions as exp
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Connection pool — reuses open TCP connections instead of reconnecting per query.
# ThreadedConnectionPool is safe for multi-threaded FastAPI + parallel eval.
# minconn=1: keep 1 connection alive (prevents Neon cold start during active use)
# maxconn=5: cap at 5 (Neon free tier allows 5 simultaneous connections)
# ---------------------------------------------------------------------------

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=5,
                    host=os.environ["POSTGRES_HOST"],
                    port=int(os.environ.get("POSTGRES_PORT", 5432)),
                    dbname=os.environ["POSTGRES_DB"],
                    user=os.environ.get("POSTGRES_READER_USER", os.environ["POSTGRES_USER"]),
                    password=os.environ.get("POSTGRES_READER_PASSWORD", os.environ["POSTGRES_PASSWORD"]),
                )
    return _pool


def _get_conn() -> psycopg2.extensions.connection:
    """Get a connection from the pool. Auto-reconnects if Neon closed it."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        conn.cursor().execute("SELECT 1")   # ping — raises if connection went stale
    except Exception:
        pool.putconn(conn, close=True)      # discard the dead connection
        conn = pool.getconn()               # open a fresh one
    return conn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_TABLES = frozenset({
    "customers",
    "support_tickets",
    "incident_logs",
    "knowledge_article_usage",
})
MAX_ROWS = 100

# Schema context injected into every SQL generation prompt.
# Exact column names and value literals from schema.sql — the LLM must
# reference these precisely or the query will fail at execution.
_SCHEMA_CONTEXT = """
customers(
    customer_id       SERIAL PRIMARY KEY,
    company_name      VARCHAR(150),
    subscription_tier VARCHAR(50),   -- 'Standard', 'Premium', 'Enterprise'
    account_status    VARCHAR(50),   -- 'Active', 'Suspended', 'Cancelled'
    sla_level         VARCHAR(50),   -- 'Standard', 'Enhanced', 'Priority'
    renewal_date      DATE,
    region            VARCHAR(100),
    created_at        TIMESTAMP
)

support_tickets(
    ticket_id       SERIAL PRIMARY KEY,
    customer_id     INTEGER REFERENCES customers(customer_id),
    issue_category  VARCHAR(100),
    severity_level  VARCHAR(50),   -- 'Low', 'Medium', 'High', 'Critical'
    ticket_status   VARCHAR(50),   -- 'Open', 'In Progress', 'Resolved', 'Closed'
    created_at      TIMESTAMP,
    resolved_at     TIMESTAMP,
    assigned_team   VARCHAR(100),
    escalation_flag BOOLEAN
)

incident_logs(
    incident_id       SERIAL PRIMARY KEY,
    incident_type     VARCHAR(100),
    severity          VARCHAR(50),   -- 'Low', 'Medium', 'High', 'Critical'
    affected_region   VARCHAR(100),
    start_time        TIMESTAMP,
    end_time          TIMESTAMP,
    resolution_status VARCHAR(50),   -- 'Open', 'In Progress', 'Resolved'
    root_cause        TEXT,
    escalation_flag   BOOLEAN,
    created_at        TIMESTAMP
)

knowledge_article_usage(
    article_id                SERIAL PRIMARY KEY,
    article_title             VARCHAR(200),
    product_version           VARCHAR(50),
    category                  VARCHAR(100),
    last_updated              DATE,
    known_issue_flag          BOOLEAN,
    internal_confidence_score FLOAT,
    created_at                TIMESTAMP
)
"""

# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------

class GeneratedQuery(BaseModel):
    sql_query:    str   # PostgreSQL SELECT statement
    query_intent: str   # what data this query retrieves (used for Langfuse logging)


# include_raw=True preserves usage_metadata for Langfuse cost tracking (SLO #14)
_llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0,
    model_kwargs={"prompt_cache_key": "langchain-prompt-caching"},
).with_structured_output(GeneratedQuery, include_raw=True)


# ---------------------------------------------------------------------------
# Step 1 — SQL generation (prompt fetched from Langfuse prompt management)
# ---------------------------------------------------------------------------

def _generate(ticket_text: str, classification: dict, prompt_obj, validation_feedback: str = "") -> tuple[GeneratedQuery, int, int]:
    prompt = prompt_obj.compile(
        schema=_SCHEMA_CONTEXT,
        max_rows=MAX_ROWS,
        ticket_text=ticket_text,
        category=classification.get("category", ""),
        severity=classification.get("severity", ""),
        reasoning=classification.get("reasoning", ""),
        validation_feedback=(
            f"IMPORTANT — previous attempt was rejected: {validation_feedback}\nGenerate a corrected query."
            if validation_feedback else ""
        ),
    )
    raw = _llm.invoke(prompt)
    if raw.get("parsing_error") or raw["parsed"] is None:
        raise ValueError(f"LLM output parsing failed: {raw.get('parsing_error')}")
    usage = (raw["raw"].usage_metadata or {})
    return raw["parsed"], usage.get("input_tokens", 0), usage.get("output_tokens", 0)


# ---------------------------------------------------------------------------
# Step 2 — SQL validation (sqlglot AST check)
# ---------------------------------------------------------------------------

def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Enforce three constraints before any query reaches the database:
      1. Must be a single SELECT statement (no DML)
      2. No DML nodes anywhere in the AST (catches DML inside subqueries)
      3. Only tables in ALLOWED_TABLES (CTE aliases excluded — they are virtual)

    Returns (is_valid, error_message).
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres")
    except Exception as e:
        return False, f"SQL parse error: {e}"

    if not isinstance(parsed, exp.Select):
        return False, f"Only SELECT allowed — got {type(parsed).__name__}"

    for node in parsed.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create)):
            return False, f"DML not allowed: {type(node).__name__}"

    # CTE aliases are virtual — exclude them from the physical table check
    cte_aliases = {cte.alias.lower() for cte in parsed.find_all(exp.CTE)}
    physical_tables = {t.name.lower() for t in parsed.find_all(exp.Table)} - cte_aliases

    disallowed = physical_tables - ALLOWED_TABLES
    if disallowed:
        return False, f"Tables not in schema: {disallowed}"

    return True, ""


# ---------------------------------------------------------------------------
# Step 3 — SQL execution
# ---------------------------------------------------------------------------

def _execute(sql: str) -> dict:
    # Uses a dedicated read-only DB user — SELECT only at the database level.
    # This is the second layer of protection after sqlglot AST validation (defense in depth).
    # SLO #12: Unauthorized Data Access = 0 violations.
    # Setup: CREATE USER support_reader WITH PASSWORD '...';
    #        GRANT SELECT ON ALL TABLES IN SCHEMA public TO support_reader;
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SET statement_timeout = '5s'")
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
        return {
            "rows":      rows,
            "row_count": len(rows),
            "truncated": len(rows) >= MAX_ROWS,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        _get_pool().putconn(conn)   # return to pool — door stays open


# ---------------------------------------------------------------------------
# Public entry point — called by Agent 3
# ---------------------------------------------------------------------------

from langfuse import observe as _observe
from src.observability.langfuse_client import get_cached_prompt


@_observe(name="agent3-sql", as_type="generation", capture_input=False, capture_output=False)
def query(ticket_text: str, classification: dict) -> dict:
    """
    LLM-to-SQL pipeline with guardrails:
      1. LLM generates a SELECT query from ticket text + classification context
      2. sqlglot validates: SELECT only, no DML, only schema tables allowed
      3. psycopg2 executes and returns rows as list of dicts

    Retry: 3 attempts with exponential backoff (1s, 2s).
    On validation failure, the error is fed back to the LLM so it can self-correct.
    Falls back to an empty result dict if all attempts fail.

    TODO Sprint 6 — Intent gating:
    If eval on TC-33 to TC-41 reveals SQL failures (wrong table selected, wrong filter
    column), add an intent detection step before SQL generation:
      Step 1: LLM classifies intent into QueryIntent (list_records / count_records /
              lookup_by_id / filter_by_status / aggregate_report) + target_table + filters
      Step 2: SQL generator receives structured QueryIntent instead of raw ticket text
    This adds 1 LLM call but gives the SQL generator a tighter, less ambiguous target.
    Only add if specific failure patterns justify the extra latency and cost.
    """
    from langfuse import observe
    from src.observability.langfuse_client import get_langfuse, calculate_cost

    lf = get_langfuse()
    last_error: Optional[Exception] = None
    validation_feedback = ""
    total_input_tokens = 0
    total_output_tokens = 0
    prompt_obj = get_cached_prompt("agent3-sql")

    lf.update_current_generation(
        model="gpt-4o-mini",
        input={"query_intent": ticket_text[:200]},
        prompt=prompt_obj,
    )

    for attempt in range(3):
        try:
            generated, in_tok, out_tok = _generate(ticket_text, classification, prompt_obj, validation_feedback)
            total_input_tokens  += in_tok
            total_output_tokens += out_tok

            is_valid, error_msg = validate_sql(generated.sql_query)
            if not is_valid:
                validation_feedback = error_msg
                last_error = ValueError(f"SQL validation: {error_msg}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                continue

            exec_result = _execute(generated.sql_query)
            result = {
                "sql_query":      generated.sql_query,
                "query_intent":   generated.query_intent,
                "rows":           exec_result["rows"],
                "row_count":      exec_result["row_count"],
                "truncated":      exec_result["truncated"],
                "error":          None,
                "_input_tokens":  total_input_tokens,
                "_output_tokens": total_output_tokens,
            }
            cost = calculate_cost(total_input_tokens, total_output_tokens)
            lf.update_current_generation(
                output={"sql_query": generated.sql_query, "row_count": exec_result["row_count"]},
                usage_details={"input": total_input_tokens, "output": total_output_tokens},
                cost_details={"total": round(cost, 6)},
                metadata={
                    "sql_query": generated.sql_query,
                    "row_count": exec_result["row_count"],
                    "truncated": exec_result["truncated"],
                },
            )
            return result

        except Exception as e:
            last_error = e
            validation_feedback = ""
            if attempt < 2:
                time.sleep(2 ** attempt)

    result = {
        "sql_query":      "",
        "query_intent":   "query failed after 3 attempts",
        "rows":           [],
        "row_count":      0,
        "truncated":      False,
        "error":          str(last_error),
        "_input_tokens":  total_input_tokens,
        "_output_tokens": total_output_tokens,
    }
    lf.update_current_generation(
        output={"error": str(last_error)},
        usage_details={"input": total_input_tokens, "output": total_output_tokens},
        cost_details={"total": round(calculate_cost(total_input_tokens, total_output_tokens), 6)},
        level="ERROR",
        status_message=str(last_error),
    )
    return result
