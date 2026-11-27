
"""
Shared pytest configuration.

Sets fake environment variables before any module-level singletons are
constructed (ChatOpenAI via src/llm.py, Langfuse).  Tests that need real LLM calls
are skipped with pytest.mark.integration.
"""

import os

os.environ.setdefault("LLM_BASE_URL",                     "https://api.openai.com/v1")
os.environ.setdefault("LLM_API_KEY",                      "test-key-00000000000000000000000")
os.environ.setdefault("POSTGRES_HOST",                    "localhost")
os.environ.setdefault("POSTGRES_PORT",                    "5432")
os.environ.setdefault("POSTGRES_DB",                      "support_intelligence")
os.environ.setdefault("POSTGRES_USER",                    "postgres")
os.environ.setdefault("POSTGRES_PASSWORD",                "password")
os.environ.setdefault("POSTGRES_READER_USER",             "support_reader")
os.environ.setdefault("POSTGRES_READER_PASSWORD",         "reader-password")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY",              "pk-lf-test-00000000")
os.environ.setdefault("LANGFUSE_SECRET_KEY",              "sk-lf-test-00000000")
os.environ.setdefault("LANGFUSE_HOST",                    "https://cloud.langfuse.com")
os.environ.setdefault("AUTH0_DOMAIN",                     "test.auth0.com")
os.environ.setdefault("AUTH0_API_AUDIENCE",               "https://test-api")
