"""
Least-privilege guard for database connections (SLO #12).

Rule: code that runs LLM-generated or user-influenced SQL must log in as the
read-only role (POSTGRES_READER_USER). Only fixed, hand-written queries may use
the owner account. These tests fail if the SQL agent's pool ever drifts back
to the owner account or silently falls back to it.

The integration test checks the real database grants and is skipped in the
normal run (`pytest -m "not integration"`); run it with `pytest -m integration`.
"""

import os
from unittest.mock import patch

import pytest

import src.rag.sql as sql


@pytest.fixture(autouse=True)
def fresh_pool():
    sql._pool = None
    yield
    sql._pool = None


def test_sql_agent_pool_logs_in_as_reader(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "neondb_owner")
    monkeypatch.setenv("POSTGRES_READER_USER", "support_reader")
    monkeypatch.setenv("POSTGRES_READER_PASSWORD", "secret")
    with patch("psycopg2.pool.ThreadedConnectionPool") as pool_cls:
        sql._get_pool()
    kwargs = pool_cls.call_args.kwargs
    assert kwargs["user"] == "support_reader"
    assert kwargs["user"] != os.environ["POSTGRES_USER"]


@pytest.mark.parametrize("missing", ["POSTGRES_READER_USER", "POSTGRES_READER_PASSWORD"])
def test_sql_agent_refuses_to_fall_back_to_owner(monkeypatch, missing):
    monkeypatch.delenv(missing, raising=False)
    with patch("psycopg2.pool.ThreadedConnectionPool") as pool_cls:
        with pytest.raises(RuntimeError, match="read-only"):
            sql._get_pool()
    pool_cls.assert_not_called()


def test_sql_agent_rejects_owner_configured_as_reader(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "neondb_owner")
    monkeypatch.setenv("POSTGRES_READER_USER", "neondb_owner")
    monkeypatch.setenv("POSTGRES_READER_PASSWORD", "secret")
    with patch("psycopg2.pool.ThreadedConnectionPool") as pool_cls:
        with pytest.raises(RuntimeError, match="read-only"):
            sql._get_pool()
    pool_cls.assert_not_called()


@pytest.mark.integration
def test_reader_role_cannot_write_in_real_database():
    from dotenv import dotenv_values
    import psycopg2

    env = dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not env.get("POSTGRES_READER_USER"):
        pytest.skip("no real .env with reader credentials")
    conn = psycopg2.connect(
        host=env["POSTGRES_HOST"], port=env.get("POSTGRES_PORT", 5432), dbname=env["POSTGRES_DB"],
        user=env["POSTGRES_READER_USER"], password=env["POSTGRES_READER_PASSWORD"],
        sslmode="require", connect_timeout=20,
    )
    try:
        cur = conn.cursor()
        for table in sorted(sql.ALLOWED_TABLES):
            cur.execute(
                "SELECT has_table_privilege(current_user, %s, 'SELECT'),"
                "       has_table_privilege(current_user, %s, 'INSERT'),"
                "       has_table_privilege(current_user, %s, 'UPDATE'),"
                "       has_table_privilege(current_user, %s, 'DELETE')",
                (table, table, table, table),
            )
            can_select, can_insert, can_update, can_delete = cur.fetchone()
            assert can_select, f"reader cannot read {table}"
            assert not (can_insert or can_update or can_delete), f"reader can write to {table}"
    finally:
        conn.close()
