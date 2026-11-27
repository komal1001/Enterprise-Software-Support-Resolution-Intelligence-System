"""
Guard: SQL text is never built from strings at runtime.

Every cursor.execute()/executemany() call must receive a fixed string literal;
values go in the separate parameters argument (`%s` placeholders). Building
the query with an f-string, .format(), % or + is how SQL injection happens.

The single allowed exception is the SQL agent's executor, which runs
LLM-generated SQL by design (ADR-002). That path is protected instead by
sqlglot validation (test_sql_safety.py), the read-only database role and
read-only transactions (test_db_least_privilege.py).
"""

import ast
import pathlib
from unittest.mock import MagicMock, patch

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCANNED_DIRS = ["src", "eval", "database", "scripts"]

# (file relative to repo root, enclosing function) allowed to execute a non-literal
ALLOWED_DYNAMIC = {("src/rag/sql.py", "_execute")}


def _visit(node, rel, enclosing):
    """Yield (file, innermost enclosing function, call) for each execute()/executemany() call."""
    for child in ast.iter_child_nodes(node):
        name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else enclosing
        if isinstance(child, ast.Call) and getattr(child.func, "attr", "") in ("execute", "executemany") and child.args:
            yield rel, enclosing, child
        yield from _visit(child, rel, name)


def _execute_calls():
    for d in SCANNED_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            yield from _visit(tree, rel, "<module>")


def _describe(arg):
    if isinstance(arg, ast.JoinedStr):
        return "f-string"
    if isinstance(arg, ast.BinOp):
        return "string built with % or +"
    if isinstance(arg, ast.Call) and getattr(arg.func, "attr", "") == "format":
        return ".format() call"
    return f"non-literal `{ast.unparse(arg)}`"


def test_scanner_finds_execute_calls():
    calls = list(_execute_calls())
    assert len(calls) >= 5, "scanner found too few execute() calls — is it still looking in the right place?"


def test_all_sql_is_a_fixed_string_literal():
    problems = []
    for rel, func_name, call in _execute_calls():
        query = call.args[0]
        if isinstance(query, ast.Constant) and isinstance(query.value, str):
            continue
        if (rel, func_name) in ALLOWED_DYNAMIC:
            continue
        problems.append(f"{rel}:{call.lineno} in {func_name}(): SQL is a {_describe(query)}")
    assert not problems, (
        "SQL must be a fixed string with values passed as parameters (%s):\n  " + "\n  ".join(problems)
    )


def test_sql_agent_executor_rolls_back_instead_of_committing():
    """A read-only query never needs COMMIT; rolling back undoes any session change it made."""
    import src.rag.sql as sql

    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = []
    with patch.object(sql, "_get_conn", return_value=conn), patch.object(sql, "_get_pool"):
        sql._execute("SELECT 1 FROM customers")
    conn.rollback.assert_called_once()
    conn.commit.assert_not_called()


def test_sql_agent_connections_are_read_only_transactions(monkeypatch):
    import src.rag.sql as sql

    monkeypatch.setattr(sql, "_pool", None)
    with patch("psycopg2.pool.ThreadedConnectionPool") as pool_cls:
        sql._get_pool()
    assert "default_transaction_read_only=on" in pool_cls.call_args.kwargs.get("options", "")
    sql._pool = None
