# AI-Powered Enterprise Email Routing & Triage System

A backend system that classifies incoming organizational emails and routes
them to the correct team, using retrieval-augmented generation (RAG) for
grounding, an LLM for classification with structured/validated output,
LangGraph for orchestration, Apache Kafka for event-driven asynchronous
processing and security screening, and a human-in-the-loop review flow for
low-confidence predictions.

**This is a portfolio prototype**, evaluated on a labeled synthetic
dataset — not a production system. See [Limitations](#limitations).

---

## Table of contents

1. [Overview & motivation](#overview--motivation)
2. [Architecture](#architecture)
3. [Data flow](#data-flow)
4. [LangGraph workflow](#langgraph-workflow)
5. [RAG architecture](#rag-architecture)
6. [Database schema](#database-schema)
7. [API documentation](#api-documentation)
8. [Email networking architecture](#email-networking-architecture)
9. [Human-in-the-loop workflow](#human-in-the-loop-workflow)
10. [Evaluation methodology & results](#evaluation-methodology--results)
11. [Limitations](#limitations)
12. [Setup instructions](#setup-instructions)
13. [Running with a real Gmail inbox](#running-with-a-real-gmail-inbox)
14. [Environment variables](#environment-variables)
15. [Docker & Kafka Architecture](#docker--kafka-architecture)
16. [Testing](#testing)
17. [Future improvements](#future-improvements)

---

## Overview & motivation

Large organizations receive emails that need to reach the right internal
team — IT, Billing, HR, Legal, and so on. Manually triaging these is slow
and error-prone. This project automates that triage: an email comes in,
the system runs automated security screening (catching credentials, OTPs,
phishing links, and dangerous attachments), retrieves relevant organizational
context via RAG, and an LLM classifies it into a department/team with a
confidence score. High-confidence predictions route automatically, while
low-confidence or flagged emails go to a human reviewer.

The system is built as an **event-driven architecture with Apache Kafka (KRaft)**:
- **FastAPI Ingestion Service**: Validates requests, stores raw emails in PostgreSQL immediately (`status=PENDING_SECURITY`), and publishes events to Kafka with zero blocking on LLM inference.
- **Security Screening Worker**: Consumes from `email-incoming`, runs regex-based security checks, updates PostgreSQL, and routes to safe (`email-approved`), review, or quarantine states.
- **Processing Worker**: Consumes from `email-approved`, applies sensitive data redaction if needed, and executes the LangGraph RAG + LLM classification workflow.
- **Streamlit UI**: Provides email triage dashboards, review queues, and analytics.
- **Standalone Fallback**: If Kafka is disabled (`KAFKA_ENABLED=false`), the system gracefully falls back to in-process FastAPI `BackgroundTasks` with inline security screening.

---

## Architecture

```mermaid
flowchart TB
    subgraph Clients
        UI[Streamlit UI]
        API_CLIENT[API Client / curl]
        MAIL_IN[Inbound Mail<br/>IMAP Fetch]
    end

    subgraph Ingestion [FastAPI Ingestion Service]
        ROUTES[API Layer<br/>/emails, /reviews, /analytics, /health]
        PRODUCER[Kafka Producer<br/>publish_email_event]
        FALLBACK[BackgroundTasks<br/>fallback if Kafka disabled]
    end

    subgraph Kafka [Apache Kafka KRaft Broker]
        T_INCOMING[Topic: email-incoming]
        T_APPROVED[Topic: email-approved]
        T_FAILED[Topic: email-failed<br/>Dead-Letter Queue]
    end

    subgraph SecurityWorker [Security Screening Worker]
        SEC_CONSUMER[Kafka Consumer<br/>group: security-worker]
        SCANNER[Security Scanner<br/>regex patterns, OTP, creds, phishing]
    end

    subgraph ProcessingWorker [Processing Worker]
        PROC_CONSUMER[Kafka Consumer<br/>group: processing-worker]
        REDACTOR[Content Redaction]
        GRAPH[LangGraph Workflow<br/>RAG retrieval + LLM classification]
    end

    DB[(PostgreSQL)]
    CHROMA[(Chroma Vector Store)]
    LLM[Groq LLM API +<br/>local embeddings]
    MAIL_OUT[Mail Server<br/>SMTP Notification]

    UI --> ROUTES
    API_CLIENT --> ROUTES
    MAIL_IN --> ROUTES

    ROUTES --> DB
    ROUTES --> PRODUCER
    PRODUCER --> T_INCOMING
    ROUTES -.KAFKA_ENABLED=false.-> FALLBACK
    FALLBACK --> GRAPH

    T_INCOMING --> SEC_CONSUMER
    SEC_CONSUMER --> SCANNER
    SEC_CONSUMER --> DB
    SEC_CONSUMER --> T_APPROVED
    SEC_CONSUMER -.exhausted retries.-> T_FAILED

    T_APPROVED --> PROC_CONSUMER
    PROC_CONSUMER --> REDACTOR
    REDACTOR --> GRAPH
    PROC_CONSUMER --> DB
    PROC_CONSUMER -.exhausted retries.-> T_FAILED

    GRAPH --> CHROMA
    GRAPH --> LLM
    GRAPH --> DB
    GRAPH -.optional.-> MAIL_OUT
```

**Layers, and why each exists:**

| Layer | Responsibility |
|---|---|
| API (`app/api/`) | Request validation (Pydantic), HTTP status codes — no business logic |
| Service (`app/services/`) | Orchestrates use cases: email ingestion, classification, routing, team notification |
| Kafka (`app/kafka/`) | KRaft event producer, consumer worker loops (`security_consumer.py`, `processing_consumer.py`), admin topic management, retry & dead-letter logic |
| Security (`app/security/`) | Deterministic regex detectors (credentials, OTPs, API keys, phishing links, dangerous file extensions), risk scoring, and LLM sensitive content redaction |
| Graph (`app/graph/`) | The actual multi-step, conditionally-branching workflow (LangGraph) |
| RAG (`app/rag/`) | Knowledge base loading, embeddings, vector store, retrieval, prompts |
| Repository (`app/database/repositories.py`) | Every direct SQL query lives here — the only file that would change if we switched ORMs |
| Integrations (`app/integrations/`) | IMAP, SMTP, PDF extraction, MIME parsing |

---

## Data flow

```mermaid
sequenceDiagram
    participant U as User / Client
    participant API as FastAPI Ingestion
    participant K as Kafka Broker
    participant SEC as Security Worker
    participant PROC as Processing Worker
    participant DB as PostgreSQL
    participant G as LangGraph (RAG+LLM)

    U->>API: POST /emails
    API->>DB: INSERT Email (status=PENDING_SECURITY)
    API->>K: publish_email_event(email-incoming)
    API-->>U: 201 Created (immediately)

    K->>SEC: consume email-incoming
    SEC->>SEC: screen_email() (credentials, phishing, extensions)
    SEC->>DB: INSERT SecurityScreening

    alt Security Approved
        SEC->>DB: UPDATE status=APPROVED
        SEC->>K: publish_email_event(email-approved)
        SEC->>K: commit offset

        K->>PROC: consume email-approved
        PROC->>DB: UPDATE status=PROCESSING
        opt Sensitive Content Found (Low Risk)
            PROC->>PROC: redact_sensitive_content()
        end
        PROC->>G: run_email_workflow()
        G->>DB: UPDATE status=ROUTED or REVIEW_REQUIRED, save RoutingResult
        PROC->>K: commit offset
    else Security Flagged (Suspicious)
        SEC->>DB: UPDATE status=SECURITY_REVIEW, INSERT Review
        SEC->>K: commit offset
    else Security Quarantined (Malicious)
        SEC->>DB: UPDATE status=QUARANTINED
        SEC->>K: commit offset
    end
```

The response to `POST /emails` returns **immediately** with `status: PENDING_SECURITY`
as soon as the email is saved to PostgreSQL and published to Kafka.
- If Kafka is unreachable during ingestion, the email is marked `KAFKA_FAILED` so it can be retried without data loss.
- If Kafka is disabled (`KAFKA_ENABLED=false`), the system executes security screening inline and schedules the workflow via FastAPI `BackgroundTasks`.

---

## LangGraph workflow

```mermaid
flowchart TD
    START([START]) --> PRE[preprocess_email]
    PRE -->|empty content| SAVE[save_result]
    PRE -->|ok| RET[retrieve_context]
    RET -->|retrieval failed| SAVE
    RET -->|ok| CLS[classify_email]
    CLS -->|invalid/ungrounded output| SAVE
    CLS -->|confidence >= threshold| ROUTE[route_email]
    CLS -->|confidence < threshold| REVIEW[human_review]
    ROUTE --> SUM[generate_summary]
    REVIEW --> SUM
    SUM --> SAVE
    SAVE --> END([END])
```

**State** (`app/graph/state.py`) is a `TypedDict` — LangGraph merges each
node's partial return into it, rather than nodes mutating a shared object.

**Nodes** (`app/graph/nodes.py`) are produced by factory functions that
close over shared dependencies (LLM client, vector store, DB session
factory), so `tests/test_graph.py` can build the exact same graph with
fake dependencies and no network calls.

**Why LangGraph** rather than a plain function pipeline: the workflow has
genuine conditional branching (confidence-based routing) driven by shared,
evolving state — exactly the case a graph-based orchestrator is for. A
plain function chain would need its own ad hoc branching logic, hand-rolled.

---

## RAG architecture

```mermaid
flowchart LR
    A[data/departments.json] -->|one Document per team| B[loader.py]
    B --> C[Local sentence-transformers<br/>embeddings, CPU]
    C --> D[(Chroma index)]
    E[Incoming email] --> F[embed query locally]
    F --> D
    D -->|top-k similar teams| G[format_context]
    G --> H[Classification prompt<br/>sent to Groq]
```

- **One document per team, not per department** — the classification
  target is team-level, so retrieval granularity matches the decision.
- **Embeddings run locally** via `sentence-transformers` (CPU), not a
  remote API — no account, no payment method, no per-call cost or
  network dependency once the model is cached. `HuggingFaceInferenceEmbeddings`
  remains in the codebase for anyone with Hugging Face billing configured.
- **The LLM never invents a `team_id`** — `classification_service.py`
  enforces that the returned `team_id` is one of the retrieved
  candidates' IDs, in code, not just via prompt instructions.
- **Why Chroma**: a local, persisted vector store is sufficient at this
  scale (10 teams); a managed vector DB would be unjustified
  infrastructure for a knowledge base this size.

---

## Database schema

```mermaid
erDiagram
    EMAIL ||--o| ROUTING_RESULT : has
    EMAIL ||--o| REVIEW : has
    EMAIL ||--o| SECURITY_SCREENING : has

    EMAIL {
        int id PK
        string sender
        string subject
        text body
        text attachment_text
        string attachment_filename
        string status
        datetime received_at
        datetime created_at
    }
    SECURITY_SCREENING {
        int id PK
        int email_id FK
        string decision
        float risk_score
        text explanation
        int detection_count
        bool sensitive_content_found
        bool phishing_signals_found
        datetime screened_at
    }
    ROUTING_RESULT {
        int id PK
        int email_id FK
        string department
        string team
        int team_id
        float confidence
        text reasoning
        text summary
        datetime routed_at
    }
    REVIEW {
        int id PK
        int email_id FK
        string predicted_team
        int predicted_team_id
        float predicted_confidence
        string final_team
        int final_team_id
        string reviewer_decision
        datetime reviewed_at
    }
```

**Complete status lifecycle with Kafka & security screening:**

* `PENDING_SECURITY`: Raw email persisted in PostgreSQL; event queued on `email-incoming`.
* `APPROVED`: Passed security checks; forwarded to `email-approved` for classification.
* `SECURITY_REVIEW`: Flagged by security scanner (suspicious links, borderline risk); held in review queue.
* `QUARANTINED`: Blocked due to malicious patterns (dangerous attachments, leaked credentials).
* `PROCESSING`: Processing worker has picked up the email and is executing the LangGraph workflow.
* `ROUTED`: High-confidence classification (`confidence >= threshold`); routed automatically.
* `REVIEW_REQUIRED`: Low-confidence classification (`confidence < threshold`); queued for human triage.
* `RESOLVED`: Human reviewer approved or corrected the classification.
* `KAFKA_FAILED`: Kafka publish failed at ingestion; safely stored in DB for recovery retry sweeps.
* `FAILED`: Workflow failure or message exhausted retries sent to dead-letter topic (`email-failed`).
* *(Legacy `PENDING` retained for backward compatibility with pre-Kafka records).*

**Why PostgreSQL + SQLAlchemy**: the data is genuinely relational — one
email has one security screening, one routing result, and sometimes one
review — a legitimate use case for a relational DB, not a default choice
made without thought.

---

## API documentation

| Method | Path | Description |
|---|---|---|
| POST | `/emails` | Submit an email; returns immediately (status `PENDING_SECURITY`), publishes to Kafka `email-incoming` (falls back to in-process background tasks if Kafka is disabled) |
| POST | `/emails/fetch` | Manually trigger an IMAP fetch of unseen emails from a real mailbox (see "Running with a real Gmail inbox" below); processes each one synchronously through the same pipeline |
| GET | `/emails` | List emails (paginated) |
| GET | `/emails/{id}` | Email detail, including routing result and security screening if available |
| GET | `/emails/{id}/classification` | Classification details (409 if not yet classified) |
| GET | `/emails/{id}/security` | Security screening details (409 if not yet screened) |
| GET | `/emails/{id}/summary` | Generated summary (409 if not yet processed) |
| GET | `/reviews/pending` | Emails awaiting human review (both low-confidence classifications and security flags) |
| POST | `/reviews/{id}` | Submit a human decision (`{"final_team": str, "final_team_id": int}`) |
| GET | `/analytics` | Aggregate stats: totals, auto-routed vs. reviewed, average confidence, by department/team |
| GET | `/health` | Liveness **and** connectivity status for both PostgreSQL and Kafka |

Full interactive docs at `/docs` once the server is running.

---

## Email networking architecture

```
Sender --SMTP--> Mail Server --IMAP--> This Application --> Classification --SMTP--> Destination Team
```

- **SMTP** (Simple Mail Transfer Protocol) is how mail gets *sent* —
  both the original sender-to-mail-server hop, and this application's
  own notification to the destination team (`app/integrations/smtp_client.py`).
- **IMAP** (Internet Message Access Protocol) is how mail gets *retrieved*
  — this application connects to a mailbox and pulls messages down
  (`app/integrations/imap_client.py`).
- **MIME** is the format email bodies/attachments are structured in;
  `app/integrations/email_parser.py` parses raw MIME bytes into
  sender/subject/body/attachments using Python's standard `email` module.
- **This project does not implement a mail server.** It's an application
  that speaks IMAP/SMTP as a *client* to existing infrastructure (Gmail,
  Outlook, or any standard mail server) — the same way any email client
  does.
- Three operating modes exist specifically so real mailbox access is
  never required to demonstrate the system: **API mode** (`POST /emails`),
  **sample dataset mode** (`data/sample_emails.json`), and **real IMAP
  mode** (calling `app.integrations.imap_client.fetch_unseen_emails()`).

---

## Human-in-the-loop workflow

1. Classification produces a `confidence` score.
2. `confidence >= CONFIDENCE_THRESHOLD` (default 0.80) leads to the `route_email` node and status `ROUTED`.
3. `confidence < CONFIDENCE_THRESHOLD` leads to the `human_review` node, status `REVIEW_REQUIRED`, and a `Review` row is created (`reviewer_decision=PENDING`).
4. A reviewer (via Streamlit's "Human Review" tab, or `POST /reviews/{id}`) sees the AI's prediction and the original email, and submits the correct team.
5. Both the **AI's original prediction** and the **human's final decision** are stored (`Review.predicted_team_id` vs. `Review.final_team_id`), with `reviewer_decision` recorded as `APPROVED` (human agreed) or `CORRECTED` (human overrode).
6. The email's status moves to `RESOLVED`.

**Important limitation, stated explicitly**: the LLM's `confidence` score
is an **application-level heuristic**, not a statistically calibrated
probability. A model reporting "0.91" is not claiming a measured 91%
historical accuracy rate at that score — it's the model's own
self-assessment, which can be miscalibrated (overconfident or
underconfident) in ways we have not independently measured. The
confidence threshold is a practical routing lever, tuned by observing
behavior, not a probabilistic guarantee.

---

## Evaluation methodology & results

Three approaches are compared on the same labeled dataset
(`data/evaluation_emails.json`, 78 synthetic examples across 10 teams,
tagged by difficulty: straightforward, reworded, ambiguous, multi-issue,
confusable):

1. **Keyword baseline** (`evaluation/keyword_baseline.py`) — no LLM, no
   embeddings, no network. Scores each team by word-overlap.
2. **LLM only, no RAG** (`evaluation/llm_only.py`) — the LLM sees only a
   bare list of team names/IDs, no descriptions or examples, and no
   retrieval narrowing.
3. **RAG + LLM** (production path, `app/services/classification_service.py`)
   — retrieval surfaces relevant team context, which grounds the LLM's
   decision and constrains it to only retrieved `team_id`s.

Run it yourself:
```bash
python -m scripts.generate_evaluation_dataset   # (re)generate the dataset
python -m evaluation.evaluate_classification    # keyword baseline only, no token needed
python -m evaluation.compare_methods            # all three approaches, needs GROQ_API_KEY
```

### Actual results from this dataset

The keyword baseline was run for real, in this repository, computing
metrics from `evaluation/metrics.py` — no numbers here are invented:

| Approach | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Keyword baseline | 82.1% | 84.4% | 81.5% | 82.4% |

**By difficulty (keyword baseline):**

| Difficulty | Accuracy |
|---|---:|
| Straightforward | 100% |
| Multi-issue | 75% |
| Confusable | 62.5% |
| Ambiguous | 60% |
| Reworded | 66.7% |

This is exactly the pattern that motivates moving beyond keyword matching:
perfect on emails phrased like the training examples, degrading sharply
once wording diverges ("reworded") or vocabulary overlaps between
similar teams ("confusable"). Semantic embeddings are specifically meant
to close that gap, since they match on meaning rather than exact words.

**LLM-only and RAG+LLM numbers are not included here** — they require a
real `GROQ_API_KEY` (free, no card needed — see Setup instructions), which
this development sandbox could not obtain interactively. Run
`python -m evaluation.compare_methods` yourself; it writes real, computed
numbers to `evaluation/comparison_report.md`. Per the project's own rule —
do not fabricate results — that comparison is left for you to actually
run rather than invented here.

### Why each major technology, backed by evidence where we have it

| Question | Answer |
|---|---|
| Why RAG? | The keyword baseline's own numbers show accuracy collapses on reworded/ambiguous phrasing — the exact failure mode semantic retrieval is meant to address. Full three-way comparison: run `compare_methods.py`. |
| Why LangGraph? | Multiple stateful steps with confidence-based conditional branching — the textbook case for a graph orchestrator over a plain function chain. |
| Why FastAPI? | Async-native, Pydantic-integrated, automatic OpenAPI docs. |
| Why PostgreSQL? | Emails, security screenings, routing results, and reviews are genuinely relational (1:1, 1:0-or-1 relationships). |
| Why Kafka & event workers? | Decouples fast HTTP ingestion from slow/flaky steps (regex security screening, local embeddings RAG, remote LLM inference, and SMTP delivery). Guarantees zero email loss via persistent KRaft topics, manual offset commits, retries with backoff, and dead-letter queueing (`email-failed`). |
| Why IMAP/SMTP? | Standard protocols for retrieving/sending mail — implemented as a client against existing infrastructure, not a custom mail server. |
| Why Chroma? | Local, zero-infra, sufficient for a 10-team knowledge base; a managed vector DB would be unjustified complexity here. |

---

## Limitations

- **Confidence is a heuristic, not a calibrated probability** (see [Human-in-the-loop workflow](#human-in-the-loop-workflow)).
- **Evaluation dataset is small and synthetic** (78 examples) — a real
  deployment would need a much larger, real-world-sourced labeled set
  before trusting these numbers operationally.
- **No OCR** — scanned/image-only PDF attachments yield no extracted text.
- **No production-scale claims** — this is a portfolio prototype. It has
  not been multi-node load-tested. However, its event-driven Kafka broker
  architecture with manual commits, retries, and dead-letter queues (`email-failed`)
  provides durability against worker process restarts. When running in the
  single-process fallback mode (`KAFKA_ENABLED=false`), in-memory `BackgroundTasks`
  do not survive API process restarts.
- **Team notification addresses are hardcoded** (`team-{id}@example.com`
  in `routing_service.py`) rather than looked up from a real directory.
- **IMAP/SMTP code is not integration-tested** against a real mail
  server, by design — only the pure MIME-parsing logic is unit tested.

---

## Setup instructions

You can run the system in three ways:
1. **Full Docker Stack (Recommended)**: Runs Postgres, KRaft Kafka, FastAPI, both Kafka workers, and Streamlit with a single command (see [Docker & Kafka Architecture](#docker--kafka-architecture)).
2. **Local Development with Kafka & Postgres in Docker**: Runs infrastructure in Docker containers and Python code on your host machine.
3. **Standalone Local Mode (No Kafka)**: Runs only Postgres in Docker with `KAFKA_ENABLED=false` for minimal resource usage.

### Option A: Local Development with Kafka (Recommended for Dev)

#### 1. Clone and configure

```bash
cp .env.example .env
```

Get a **free Groq API key** (no credit card required) at
https://console.groq.com/keys and set `GROQ_API_KEY` in `.env`. This
powers classification and summarization.

Configure Kafka settings in `.env` for host connectivity:
```env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_ENABLED=true
```

Embeddings run **locally** via `sentence-transformers` — no API key, no
account, no cost. The first run downloads the model (~90MB) and caches
it locally.

> **Why not Hugging Face for everything?** HF's Inference Providers now
> require a payment method on file even to use free-tier credits — a real
> barrier for a portfolio project with no budget. Groq's free tier
> requires no card at all, and running embeddings locally sidesteps the
> issue entirely for that piece. The `HuggingFaceLLMClient` /
> `HuggingFaceInferenceEmbeddings` classes are still in the codebase if
> you do have HF billing set up and want to use them instead — see
> `app/core/dependencies.py`.

#### 2. Start Postgres and Kafka

```bash
docker compose up -d db kafka
```

#### 3. Install dependencies

Requires **Python 3.13** (3.12+ also supported — every dependency
publishes compatible wheels for either).

> **Windows note**: `psycopg2-binary` only ships prebuilt Python 3.13
> wheels from version `2.9.10` onward — `requirements.txt` is pinned to
> `2.9.12` specifically for this reason.

> **Optional — smaller download**: `pip install -r requirements-dev.txt`
> pulls the standard `torch` wheel, which bundles full CUDA support
> (~2GB). If you'd rather save bandwidth on CPU-only machines:
> ```bash
> pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
> ```

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

#### 4. Build the RAG index

```bash
python -m scripts.build_index
```

#### 5. Start Application Processes (in separate terminals)

* **Terminal 1 (FastAPI API):**
  ```bash
  uvicorn app.main:app --reload
  ```
  Visit http://localhost:8000/docs.

* **Terminal 2 (Security Screening Worker):**
  ```bash
  python -m app.kafka.security_consumer
  ```

* **Terminal 3 (Processing Worker):**
  ```bash
  python -m app.kafka.processing_consumer
  ```

* **Terminal 4 (Streamlit UI):**
  ```bash
  streamlit run streamlit_app/app.py
  ```
  Visit http://localhost:8501.

---

### Option B: Standalone Local Mode (Without Kafka)

If you do not want to run Kafka locally:
1. Start Postgres only: `docker compose up -d db`
2. In `.env`, set: `KAFKA_ENABLED=false`
3. Terminal 1: `uvicorn app.main:app --reload`
4. Terminal 2: `streamlit run streamlit_app/app.py`

In this mode, security screening runs synchronously inline, and classification executes via FastAPI `BackgroundTasks`.

---

### Trying the Application

- **Mode 1 (API/demo)**: use the Streamlit "Submit Email" tab, or:
  ```bash
  curl -X POST localhost:8000/emails -d '{"sender":"user@example.com","subject":"VPN help","body":"I cannot log into the VPN"}' -H "Content-Type: application/json"
  ```
- **Mode 2 (sample data)**: click "Load sample emails" in Streamlit, which reads `data/sample_emails.json`.
- **Mode 3 (real IMAP)**: set `IMAP_HOST`/`IMAP_USER`/`IMAP_PASSWORD` in `.env`, then trigger `POST /emails/fetch` or use the Streamlit interface.

---

## Running with a real Gmail inbox

Everything above ("Setup instructions") describes **synthetic/API
testing**: emails you submit yourself via `POST /emails`, Streamlit's
"Load sample emails" button, or `scripts/bulk_submit.py`. This section
covers **real Gmail integration** — an actual mailbox, real incoming
mail, fetched via IMAP and forwarded via SMTP.

Both paths converge on the exact same pipeline: `POST /emails` and
`POST /emails/fetch` both end up calling `process_email()` — nothing
about classification, RAG, or the LangGraph workflow differs between a
synthetic test email and a real one from Gmail.

```
Synthetic/API testing                    Real Gmail integration
──────────────────────                   ──────────────────────
POST /emails  ──┐                        Gmail inbox ──IMAP──┐
                ├──▶ process_email() ◀───────────────────────┘
Streamlit UI  ──┘         │
                          ▼
              LangGraph (RAG + LLM classification)
                          │
                          ▼
                    PostgreSQL
                          │
                          ▼ (if ROUTED)
                    SMTP forward to destination team
```

### 1. Create a Gmail App Password

Gmail will reject IMAP/SMTP login with your normal password if 2-Step
Verification is enabled (which Google requires for most accounts). Use
an **App Password** instead:

1. Enable 2-Step Verification: https://myaccount.google.com/security
2. Create an App Password: https://myaccount.google.com/apppasswords
3. Use that 16-character password as `IMAP_PASSWORD`/`SMTP_PASSWORD` below — not your real Gmail password.

### 2. Add IMAP/SMTP variables to `.env`

```env
IMAP_HOST=imap.gmail.com
IMAP_PORT=993
IMAP_USER=your-address@gmail.com
IMAP_PASSWORD=your-16-char-app-password

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-address@gmail.com
SMTP_PASSWORD=your-16-char-app-password
```

(These use the same Gmail address/App Password for both — Gmail's IMAP
and SMTP servers share credentials.)

### 3. Start the application as usual

```bash
uvicorn app.main:app --reload
```

No separate process, worker, or service is needed for IMAP — it's just
another endpoint on the same FastAPI app.

### 4. Send a test email to your configured Gmail inbox

From any other email account, send something like:

> Subject: Can't access my account
> Body: I've been trying to log in for the last hour with no luck.

### 5. Trigger the fetch

```bash
curl -X POST http://localhost:8000/emails/fetch
```

or from Streamlit/`/docs`, call `POST /emails/fetch`.

This is a **manual, one-shot fetch** — not continuous polling. Call it
again whenever you want to check for new mail. (Adding a scheduler would
be a reasonable future addition, but isn't needed to demonstrate the
integration and was deliberately left out — see "What this doesn't do"
below.)

### 6. What happens

1. The app connects to `imap.gmail.com`, searches for `UNSEEN` messages.
2. Each one is fetched via `BODY.PEEK[]` — this does **not** mark it
   `\Seen` by itself, unlike a plain fetch.
3. It's parsed with the same `parse_raw_email()` used everywhere else
   (`app/integrations/email_parser.py`) — sender, subject, body,
   attachments.
4. It's handed to `process_parsed_email_sync()`, which creates an `Email`
   row and calls the same `process_email()` that `POST /emails` uses:
   RAG retrieval → LLM classification → confidence check → save.
5. If that completes without a fatal error, the message is marked
   `\Seen`, so it won't be re-fetched next time. (If classification
   itself fails, e.g. bad LLM output, the email is still marked `\Seen`
   — a `FAILED` status is a legitimate recorded outcome, not something
   retrying the *same* IMAP fetch would fix. Only a genuine failure to
   even save the email, like the database being down, leaves it unseen
   for retry.)
6. If the result is `ROUTED` (high confidence), `save_result` calls
   `notify_destination_team()`, which sends an email via SMTP to
   `team-{id}@example.com` containing the original sender, subject,
   body, predicted team, and generated summary. (That address template
   is a placeholder — see `app/services/routing_service.py` — replace it
   with real team addresses for actual use.)
7. If the result is `REVIEW_REQUIRED` (low confidence), it is **not**
   forwarded via SMTP yet — it waits in `GET /reviews/pending` for a
   human decision via `POST /reviews/{id}`, same as any API-submitted
   email.
8. Check the result the same way you would for any email:
   `GET /emails`, `GET /emails/{id}`, or the Streamlit UI.

### What this doesn't do (by design)

- **No continuous polling.** `POST /emails/fetch` is triggered manually.
  A scheduler (cron, APScheduler, etc.) calling this endpoint
  periodically would be a small, separate addition — not built here to
  keep the surface area minimal, per this feature's own scope.
- **No new classification logic.** `app/graph`, `app/rag`, and
  `app/services/classification_service.py` are completely untouched by
  this change. IMAP-sourced and API-sourced emails are indistinguishable
  once they're an `Email` row in the database.
- **No message queue, worker process, or scheduler infrastructure** —
  `POST /emails/fetch` runs the fetch loop synchronously, in-request,
  using the existing FastAPI process.

---

## Environment variables

See `.env.example` for the full list with inline comments.

| Variable | Default / Example | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/email_router` | PostgreSQL connection string |
| `GROQ_API_KEY` | *(Required for LLM)* | API key from https://console.groq.com/keys (powers classification and summary) |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` (Docker) or `localhost:9092` (local host dev) | Apache Kafka broker address |
| `KAFKA_ENABLED` | `true` | When `false`, disables Kafka and runs inline security screening + background tasks |
| `CONFIDENCE_THRESHOLD` | `0.80` | Threshold above which emails route automatically |
| `IMAP_HOST` / `IMAP_USER` / `IMAP_PASSWORD` | *(Optional)* | Inbound credentials for real mailbox fetching |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` | *(Optional)* | Outbound credentials for team notifications |
| `API_BASE_URL` | `http://localhost:8000` | Base URL used by the Streamlit frontend |

---

## Docker & Kafka Architecture

The full stack runs via Docker Compose with KRaft-based Apache Kafka (no ZooKeeper required):

```bash
docker compose up --build
```

### Services

| Service | Container / Port | Description |
|---|---|---|
| `db` | `postgres:16` (5432) | PostgreSQL persistence for emails, routing results, reviews, and security screenings |
| `kafka` | `apache/kafka:3.8.0` (9092, 29092) | Single-node KRaft broker (`kafka:29092` internal, `localhost:9092` external host) |
| `api` | FastAPI (8000) | Ingestion API (`POST /emails`), publishes to Kafka, falls back to `KAFKA_FAILED` if down |
| `security-worker` | Python worker | Consumes `email-incoming`, screens content, routes to `email-approved`, `SECURITY_REVIEW`, or `QUARANTINED` |
| `processing-worker` | Python worker | Consumes `email-approved`, runs LangGraph RAG + LLM classification |
| `streamlit` | Streamlit (8501) | Triage dashboard, review queue, and analytics |

### Verifying Kafka and Event Flow

1. **Verify Kafka Broker is running:**
   ```powershell
   docker compose ps kafka
   ```
2. **List Kafka Topics:**
   ```powershell
   docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
   ```
   Expected topics: `email-incoming`, `email-approved`, `email-failed`.
3. **Follow Consumer Worker Logs:**
   ```powershell
   docker compose logs -f security-worker processing-worker
   ```
4. **Inspect Service Health:**
   ```powershell
   curl http://localhost:8000/health
   ```
   Returns `{"status": "ok", "app_env": "development", "database": "connected", "kafka": "connected"}`.

---

## Testing

```bash
pytest tests/ -v
```

85 tests, **zero external API dependency, zero network calls**:

| File | Covers |
|---|---|
| `test_security.py` | Regex pattern detectors (OTPs, credentials, API keys, phishing URLs, malicious extensions), risk scoring, quarantine rules, and LLM content redaction |
| `test_kafka_integration.py` | Producer serialization, failure timeouts, security consumer decisions, idempotency, and admin topic management |
| `test_health.py` | App boot, `/health` shape including database and Kafka connectivity reporting |
| `test_rag.py` | Loader, vector store, retriever (fake embeddings) |
| `test_classification.py` | Structured output validation, retry logic, team_id grounding guardrail (fake LLM) |
| `test_graph.py` | All LangGraph branches: routed, review-required, failed (fake LLM + fake embeddings + SQLite) |
| `test_database.py` | Repository CRUD, relationships, analytics aggregation (SQLite) |
| `test_api.py` | Full request -> background task -> DB -> response cycle for every endpoint (SQLite + fakes) |
| `test_email_parser.py` | MIME parsing, attachment size limits, filename sanitization |
| `test_pdf_parser.py` | PDF text extraction, including per-page failure handling |
| `test_evaluation.py` | Metric computation correctness, keyword baseline logic |


External-service code (real Groq API calls, real IMAP/SMTP
connections) is deliberately **not** unit tested — see
[Limitations](#limitations) — since doing so would make the suite
flaky and dependent on credentials/network, which the project's own
rules rule out.

---

## Future improvements

- Larger, more rigorously validated evaluation dataset (target 200-300 examples, ideally including some real anonymized data)
- Multi-partition Kafka topic scaling with dynamic consumer group rebalancing
- OpenTelemetry distributed tracing across the Kafka event pipeline (`api` -> `security-worker` -> `processing-worker`)
- OCR for scanned PDF attachments
- Real team-directory lookup instead of a hardcoded notification address template
- Calibration study for the confidence score (e.g. temperature scaling against a held-out labeled set) rather than treating it as a raw heuristic
- Swap-in OpenAI/Ollama LLM client (the `LLMClient` abstraction already supports this — just a new subclass)
