-- Enterprise Support Intelligence System — PostgreSQL Schema
-- Run this once to create all tables before running generate_data.py

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE customers (
    customer_id     SERIAL PRIMARY KEY,
    company_name    VARCHAR(150) NOT NULL,
    subscription_tier VARCHAR(50),
    account_status  VARCHAR(50),
    sla_level       VARCHAR(50),
    renewal_date    DATE,
    region          VARCHAR(100),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE support_tickets (
    ticket_id       SERIAL PRIMARY KEY,
    customer_id     INTEGER REFERENCES customers(customer_id) ON DELETE CASCADE,
    issue_category  VARCHAR(100),
    severity_level  VARCHAR(50),
    ticket_status   VARCHAR(50),
    created_at      TIMESTAMP,
    resolved_at     TIMESTAMP,
    assigned_team   VARCHAR(100),
    escalation_flag BOOLEAN DEFAULT FALSE
);

CREATE TABLE incident_logs (
    incident_id       SERIAL PRIMARY KEY,
    incident_type     VARCHAR(100),
    severity          VARCHAR(50),
    affected_region   VARCHAR(100),
    start_time        TIMESTAMP,
    end_time          TIMESTAMP,
    resolution_status VARCHAR(50),
    root_cause        TEXT,
    escalation_flag   BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE knowledge_article_usage (
    article_id               SERIAL PRIMARY KEY,
    article_title            VARCHAR(200),
    product_version          VARCHAR(50),
    category                 VARCHAR(100),
    last_updated             DATE,
    known_issue_flag         BOOLEAN,
    internal_confidence_score FLOAT,
    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector table for RAG document chunks (pgvector)
-- Table schema is managed by LlamaIndex PGVectorStore (VectorStoreIndex).
-- LlamaIndex creates this table automatically during ingestion (src/retrieval/rag.py).
-- Columns: id (uuid), text, metadata_ (jsonb), node_id, embedding vector(1536)
-- doc_title and page_ref are stored inside metadata_ as file_name and page_label.
