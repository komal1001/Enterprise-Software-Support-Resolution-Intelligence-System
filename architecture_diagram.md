```mermaid
graph LR
    %% 🖥️ Client Layer
    subgraph Client_Layer ["🖥️ CLIENT LAYER"]
        UI["🌐 React UI (Vite)<br>• Chat Interface<br>• Real-Time Progress Panel"]
    end

    %% 🛡️ API Gateway & Security Layer
    subgraph Gateway_Layer ["🛡️ API GATEWAY & SECURITY"]
        API["⚡ FastAPI App<br>(Uvicorn Engine)"]
        Auth0["🔐 Auth0 Security<br>JWT & RBAC Filters"]
        Guard["🚫 Input Guardrails<br>PII & Prompt Injection Blocks"]
    end

    %% ⚙️ Core Processing Layer
    subgraph Core_Layer ["⚙️ LANGGRAPH ORCHESTRATION ENGINE"]
        State["🧠 StateGraph Context<br>(MemorySaver Session Store)"]
        Synth["📝 Response Synthesizer<br>(Server-Sent Events)"]
    end

    %% 💾 Storage & Metrics Layer
    subgraph Storage_Layer ["💾 ENTERPRISE DATA & TELEMETRY LAYER"]
        Neon["🐘 Neon PostgreSQL DB<br>Structured Customer Data"]
        PgV["🧬 pgvector Cache / Index<br>155 Hybrid RAG Nodes"]
        Lfuse["📊 Langfuse Trace Core<br>Costs & RAGAS Evaluator"]
    end

    %% --- LEFT TO RIGHT DATA FLOWS ---
    UI ===>|1. Post Ticket| API
    API ===>|2. Verify Identity| Auth0
    Auth0 ===>|3. Enforce Policy| Guard
    Guard ===>|4. Validated Payload| State
    State ===>|5. Routing Decision| Synth
    State <.===>|Read/Write State| Neon
    State <.===>|Matches & Chunks| PgV
    Synth ===>|6. Push Tokens via SSE| UI
    Synth ===>|7. Async Trace Dump| Lfuse

    %% --- INLINE UNIVERSAL COLOR STYLING ---
    style UI fill:#38bdf8,stroke:#0284c7,stroke-width:2px,color:#000
    
    style API fill:#fb7185,stroke:#e11d48,stroke-width:2px,color:#000
    style Auth0 fill:#fb7185,stroke:#e11d48,stroke-width:2px,color:#000
    style Guard fill:#fb7185,stroke:#e11d48,stroke-width:2px,color:#000
    
    style State fill:#c084fc,stroke:#9333ea,stroke-width:2px,color:#000
    style Synth fill:#c084fc,stroke:#9333ea,stroke-width:2px,color:#000
    
    style Neon fill:#34d399,stroke:#059669,stroke-width:2px,color:#000
    style PgV fill:#34d399,stroke:#059669,stroke-width:2px,color:#000
    style Lfuse fill:#facc15,stroke:#ca8a04,stroke-width:2px,color:#000

    %% Line Colors
    linkStyle 0 stroke:#38bdf8,stroke-width:3px;
    linkStyle 1 stroke:#fb7185,stroke-width:3px;
    linkStyle 2 stroke:#fb7185,stroke-width:3px;
    linkStyle 3 stroke:#c084fc,stroke-width:3px;
    linkStyle 4 stroke:#c084fc,stroke-width:3px;
    linkStyle 5 stroke:#34d399,stroke-width:2px,stroke-dasharray: 3 3;
    linkStyle 6 stroke:#34d399,stroke-width:2px,stroke-dasharray: 3 3;
    linkStyle 7 stroke:#38bdf8,stroke-width:3px;
    linkStyle 8 stroke:#facc15,stroke-width:3px;