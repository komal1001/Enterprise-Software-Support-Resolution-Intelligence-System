# High-Level Flow Architecture Diagram
# Enterprise Software Support & Resolution Intelligence System

```mermaid
flowchart TD

    %% ─────────────────────────────────────────
    %% LAYER 1 · INPUT
    %% ─────────────────────────────────────────
    USER([👤 End User\nPortal / Email / API])
    USER -->|Support ticket / query| API_IN

    subgraph L1["🌐 Layer 1 — API Layer (FastAPI)"]
        API_IN[Request Ingestion]
        API_IN --> AUTH[Authentication &\nRole-Based Access Control]
        AUTH --> VAL[Input Validation &\nRate Limiting]
    end

    VAL --> ICA

    %% ─────────────────────────────────────────
    %% LAYER 2 · AGENT ORCHESTRATION (LangGraph / CrewAI)
    %% ─────────────────────────────────────────
    subgraph L2["🧠 Layer 2 — Agent Orchestration (LangGraph / CrewAI)"]
        direction TB

        ICA["🔍 Intent Classification Agent\n─────────────────────────────\nCategory: usage · integration\n         incident · billing\nSeverity: Low · Medium · High · Critical\nMaintains multi-turn context"]

        ICA --> ROUTER{{"⚡ Intelligent\nQuery Router"}}

        ROUTER -->|"55% — Docs / Usage / Config"| DRA
        ROUTER -->|"30% — Account / Billing / Sub"| AVA
        ROUTER -->|"15% — Hybrid"| HYB
        ROUTER -->|"Severity = High / Critical"| ISAA

        DRA["📚 Documentation\nRetrieval Agent\n────────────────\nSemantic search\nover knowledge base\nSource attribution"]

        AVA["🗄️ Account Validation\nAgent\n────────────────\nSubscription status\nAccount flags\nIncident history"]

        HYB["🔀 Hybrid Coordinator\n────────────────\nRAG + SQL reasoning\nCross-reference\nvalidation"]

        ISAA["🚨 Incident Severity\nAssessment Agent\n────────────────\nOutage detection\nSecurity scan\nPattern analysis"]

        DRA --> SYNTH
        AVA --> SYNTH
        HYB --> SYNTH
        ISAA --> MULTI

        subgraph MULTI["⚙️ Multi-Agent Validation — Plan → Act → Check"]
            direction LR
            SEV[Severity\nRe-evaluator] --> CORR[Evidence\nCorroborator] --> CONF[Confidence\nRecalculator]
        end

        SYNTH["📝 Response Synthesizer\n────────────────────────\n✔ Troubleshooting steps\n✔ Source / doc links\n✔ Confidence score\n✔ Severity label\n✔ SQL validation output"]

        MULTI --> ESC_DEC
        SYNTH --> ESC_DEC

        ESC_DEC{{"🔀 Escalation\nDecision Gate"}}
    end

    %% ─────────────────────────────────────────
    %% LAYER 3 · KNOWLEDGE & RETRIEVAL
    %% ─────────────────────────────────────────
    subgraph L3["📦 Layer 3 — Retrieval & Knowledge Layer"]
        direction LR

        VEC[("🗂️ Vector Store\n─────────────\nProduct docs\nTroubleshooting guides\nKB articles\nAPI manuals\n[pgvector / AI Search]")]

        PG[("🐘 PostgreSQL\n─────────────\ncustomers\nsupport_tickets\nincident_logs\nknowledge_article_usage")]
    end

    DRA <-->|Semantic search| VEC
    AVA <-->|SQL queries| PG
    HYB <-->|Both| VEC
    HYB <-->|Both| PG
    ISAA <-->|Incident logs| PG

    %% ─────────────────────────────────────────
    %% LAYER 4 · EXTERNAL TOOLS
    %% ─────────────────────────────────────────
    subgraph L4["🔌 Layer 4 — External Tools (MCP Integrations)"]
        MCP1[Ticketing System\nConnector]
        MCP2[Notification /\nAlert Service]
        MCP3[CRM / Billing\nSystem]
    end

    L2 <-.->|MCP tool calls| L4

    %% ─────────────────────────────────────────
    %% LAYER 5 · HUMAN-IN-THE-LOOP ESCALATION
    %% ─────────────────────────────────────────
    ESC_DEC -->|"✅ Confidence >= threshold\nSeverity <= High"| AUTO_RESP
    ESC_DEC -->|"🚨 Critical / Low-confidence\nOutage / Security / Data loss\nSystemic pattern / User request"| EMA

    AUTO_RESP["✅ Automated Response\nto User\n(steps · sources · score · severity)"]
    AUTO_RESP --> USER

    subgraph L5["🚒 Layer 5 — Human-in-the-Loop (Escalation)"]
        EMA["🚒 Escalation Manager\nAgent"]
        EMA --> CTX["📋 Context Package\n──────────────────────\n• Full ticket history\n• Retrieved documentation\n• Structured account data\n• Incident logs\n• Agent reasoning trace"]
        CTX --> HUMAN(["👨‍💼 Human Support\nEngineer / L2 / Engineering"])
    end

    HUMAN -->|Resolution or feedback| USER

    %% ─────────────────────────────────────────
    %% LAYER 6 · OBSERVABILITY
    %% ─────────────────────────────────────────
    subgraph L6["📊 Layer 6 — Observability & SLO (Langfuse)"]
        direction LR
        OBS1["📈 SLO Dashboard\n──────────────────\nTSR >= 90%\nP95 Latency <= 3-6s\nSQL Correctness >= 95%\nMisclassification < 3%\nCost per ticket"]
        OBS2["🔎 Distributed Traces\nAgent reasoning steps\nRetrieval latency\nTool call logs"]
        OBS3["🗃️ Audit Log\nDecision records\nEscalation events\nSource citations"]
    end

    L2 -->|Metrics & traces| L6
    L5 -->|Escalation events| L6

    %% ─────────────────────────────────────────
    %% HIGH-RISK TRIGGER ANNOTATIONS
    %% ─────────────────────────────────────────
    ISAA -. "Triggers multi-agent if:\n• Production outage\n• Security vulnerability\n• Data loss complaint\n• Systemic failure pattern\n• Unresolved critical alert\n• Conflicting doc versions" .-> MULTI

    %% ─────────────────────────────────────────
    %% STYLES
    %% ─────────────────────────────────────────
    classDef agent    fill:#2E86C1,stroke:#1a5276,color:#fff
    classDef router   fill:#E67E22,stroke:#9C5A0A,color:#fff
    classDef store    fill:#1E8449,stroke:#145A32,color:#fff
    classDef escalate fill:#C0392B,stroke:#7B241C,color:#fff
    classDef ok       fill:#7D3C98,stroke:#4A235A,color:#fff
    classDef obs      fill:#117A65,stroke:#0A4F42,color:#fff
    classDef mcp      fill:#626567,stroke:#2C3E50,color:#fff

    class ICA,DRA,AVA,ISAA,HYB,SEV,CORR,CONF,EMA agent
    class ROUTER,ESC_DEC router
    class VEC,PG store
    class HUMAN,CTX escalate
    class AUTO_RESP,SYNTH ok
    class OBS1,OBS2,OBS3 obs
    class MCP1,MCP2,MCP3 mcp
```

---

## System Layer Summary

| # | Layer | Technology | Role |
|---|---|---|---|
| 1 | **API** | FastAPI | Auth, RBAC, rate limiting, input validation |
| 2 | **Orchestration** | LangGraph / CrewAI | 5 agents + router + Plan→Act→Check loop |
| 3 | **Knowledge** | pgvector + PostgreSQL | RAG retrieval + structured SQL queries |
| 4 | **External Tools** | MCP Integrations | Ticketing, CRM, notification connectors |
| 5 | **Escalation** | Human-in-the-Loop | Escalation Manager → full context handoff |
| 6 | **Observability** | Langfuse | SLO tracking, traces, audit logs |

---

## Query Routing Logic

```
Incoming query
│
├── 55% — Usage / Config / Docs     → Documentation Retrieval Agent  → Vector Store (RAG)
├── 30% — Account / Billing / Sub   → Account Validation Agent       → PostgreSQL (SQL)
├── 15% — Hybrid signals            → Hybrid Coordinator             → RAG + SQL
└── Severity = High / Critical      → Incident Severity Assessment   → Multi-Agent Validation
```

---

## PostgreSQL Tables

| Table | Used By |
|---|---|
| `customers` | Account Validation Agent |
| `support_tickets` | Intent Classifier, Escalation Manager |
| `incident_logs` | Incident Severity Agent, Multi-Agent Validation |
| `knowledge_article_usage` | Documentation Retrieval Agent, Observability |

---

## Escalation Trigger Conditions

| Trigger | Action |
|---|---|
| Severity = Critical | Force multi-agent validation → escalate |
| Confidence score < threshold | Escalate with context package |
| Production outage detected | Immediate escalation |
| Security vulnerability found | Immediate escalation |
| Data loss complaint, no logs | Escalate |
| Multiple tickets = systemic failure | Escalate |
| User explicitly requests human | Escalate |

---

## SLO Targets

| Metric | Target |
|---|---|
| Task Success Rate (TSR) | >= 90% |
| P95 Response Latency | <= 3–6 seconds |
| SQL Query Correctness | >= 95% |
| Critical Misclassification Rate | < 3% |
| Cost per Ticket | Within defined budget |
