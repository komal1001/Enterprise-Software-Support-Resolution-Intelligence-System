"""
Unit tests for SQL safety layer in src/retrieval/sql.py.

validate_sql() is a pure function — no DB, no LLM.
Tests prove ADR-002: sqlglot rejects all non-SELECT statements.
"""

import pytest
from src.retrieval.sql import validate_sql, ALLOWED_TABLES


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
