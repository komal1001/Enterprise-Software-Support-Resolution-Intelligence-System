
"""
Shared pytest configuration.

Sets fake environment variables before any module-level singletons are
constructed (AzureChatOpenAI, Langfuse).  Tests that need real Azure calls
are skipped with pytest.mark.integration.
"""

import os

os.environ.setdefault("AZURE_OPENAI_ENDPOINT",            "https://test.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT",          "gpt-4o-mini")
os.environ.setdefault("AZURE_OPENAI_API_VERSION",         "2024-02-01")
os.environ.setdefault("AZURE_OPENAI_API_KEY",             "test-key-00000000000000000000000")
os.environ.setdefault("AZURE_OPENAI_EMBEDDING_DEPLOYMENT","text-embedding-3-small")
os.environ.setdefault("POSTGRES_HOST",                    "localhost")
os.environ.setdefault("POSTGRES_PORT",                    "5432")
os.environ.setdefault("POSTGRES_DB",                      "support_intelligence")
os.environ.setdefault("POSTGRES_USER",                    "postgres")
os.environ.setdefault("POSTGRES_PASSWORD",                "password")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY",              "pk-lf-test-00000000")
os.environ.setdefault("LANGFUSE_SECRET_KEY",              "sk-lf-test-00000000")
os.environ.setdefault("LANGFUSE_HOST",                    "https://cloud.langfuse.com")
os.environ.setdefault("AUTH0_DOMAIN",                     "test.auth0.com")
os.environ.setdefault("AUTH0_API_AUDIENCE",               "https://test-api")
