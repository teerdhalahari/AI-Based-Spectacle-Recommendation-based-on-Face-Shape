# Face2Frame – AI Based Spectacle Style Advisor  
## Production-Ready Academic Project Architecture

---

## 1. Folder Structure

```
Face2FrameCursor/
├── app/
│   ├── __init__.py                 # Flask app factory, extensions init
│   ├── config.py                   # Environment-based configuration
│   │
│   ├── models/                     # MVC – Models (data layer)
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── session.py
│   │   ├── face_analysis.py
│   │   ├── spectacle.py
│   │   └── recommendation.py
│   │
│   ├── views/                      # MVC – Views (presentation / API endpoints)
│   │   ├── __init__.py
│   │   ├── auth.py                 # Login, register, logout, profile
│   │   ├── upload.py               # Image upload & validation
│   │   ├── analysis.py             # Face analysis triggers & results
│   │   ├── recommendation.py       # Spectacle recommendations API
│   │   └── health.py               # Health check, readiness
│   │
│   ├── controllers/                # MVC – Controllers / business logic
│   │   ├── __init__.py
│   │   ├── auth_controller.py
│   │   ├── image_controller.py
│   │   ├── face_controller.py
│   │   └── recommendation_controller.py
│   │
│   ├── services/                   # Domain services (reusable logic)
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── image_validation_service.py
│   │   ├── face_analysis_service.py
│   │   └── recommendation_service.py
│   │
│   ├── pipelines/                  # Multi-step processing pipelines
│   │   ├── __init__.py
│   │   ├── image_validation_pipeline.py
│   │   └── analysis_pipeline.py    # Validate → Analyze → Recommend
│   │
│   ├── ml/                         # ML / AI modules (face & recommendation)
│   │   ├── __init__.py
│   │   ├── face_detector.py
│   │   ├── face_analyzer.py        # Shape, features, landmarks
│   │   ├── spectacle_catalog.py    # Spectacle metadata & embeddings
│   │   └── recommender.py          # Matching logic (face shape → frames)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── decorators.py           # @login_required, @validate_image, etc.
│   │   ├── exceptions.py           # Custom exceptions
│   │   └── helpers.py              # File I/O, hashing, response helpers
│   │
│   └── static/                     # Static assets (optional)
│       ├── css/
│       ├── js/
│       └── uploads/                # Temporary/user uploads (configurable path)
│
├── migrations/                     # DB migrations (e.g. Flask-Migrate)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/                           # API docs, runbooks
├── scripts/                        # Seed DB, train models, one-off tasks
├── instance/                       # Instance-specific config & SQLite DB
│   └── face2frame.db
├── requirements.txt
├── .env.example
├── run.py                          # Entry point (development)
├── wsgi.py                         # Production WSGI entry
└── ARCHITECTURE.md                 # This document
```

---

## 2. Module Responsibilities

| Layer / Module | Responsibility |
|----------------|----------------|
| **app/__init__.py** | Create Flask app, register blueprints, init DB and extensions (e.g. LoginManager), load config. |
| **app/config.py** | Central config: `SECRET_KEY`, `SQLALCHEMY_DATABASE_URI`, upload limits, allowed MIME types, paths, feature flags. |
| **models/** | SQLAlchemy (or similar) ORM: User, Session/Token, FaceAnalysis, Spectacle, Recommendation. Define schema and relationships only; no business logic. |
| **views/auth.py** | HTTP endpoints: register, login, logout, profile, change password. Parse request, call controller, return JSON/HTML. |
| **views/upload.py** | Endpoints: upload image, get upload status. Delegates validation and storage to controller/services. |
| **views/analysis.py** | Endpoints: start analysis, get analysis result by ID. Thin layer over controllers. |
| **views/recommendation.py** | Endpoints: get recommendations for a face analysis, list history. |
| **views/health.py** | Health/readiness: DB connectivity, optional dependency checks. |
| **controllers/** | Orchestrate use cases: call services, pipelines, and models; handle transactions and errors; return DTOs for views. |
| **services/auth_service.py** | Hash/verify passwords, create/revoke sessions, token handling, user lookup. |
| **services/image_validation_service.py** | Validate file type, size, dimensions; optional basic content checks (e.g. not empty image). |
| **services/face_analysis_service.py** | Run face detection and analysis (via ml/), persist FaceAnalysis records. |
| **services/recommendation_service.py** | Generate spectacle recommendations from face analysis (via ml/recommender), persist Recommendation records. |
| **pipelines/image_validation_pipeline.py** | Sequential steps: receive file → validate → optionally store/save path → return result (success/failure + messages). |
| **pipelines/analysis_pipeline.py** | End-to-end: validated image → face analysis → recommendation; coordinate services and persist results. |
| **ml/face_detector.py** | Detect face(s) in image; return bounding box(es) or failure. |
| **ml/face_analyzer.py** | From cropped face: extract shape, landmarks, features (e.g. face shape class). |
| **ml/spectacle_catalog.py** | Load and expose spectacle metadata (and optional embeddings) for recommendation. |
| **ml/recommender.py** | Map face shape/features to spectacle IDs; scoring and filtering logic. |
| **utils/decorators.py** | `@login_required`, `@validate_image`, rate limit, etc. |
| **utils/exceptions.py** | Custom exceptions (e.g. ValidationError, FaceNotFoundError) for consistent error handling. |
| **utils/helpers.py** | Safe file save, path generation, response formatting. |

---

## 3. Data Flow Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (Browser / API Client)                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
            ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
            │  Auth Views   │     │ Upload Views  │     │ Analysis /    │
            │  (login,      │     │ (POST image)  │     │ Recommendation│
            │   register)   │     │               │     │ Views         │
            └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
                    │                     │                     │
                    ▼                     ▼                     ▼
            ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
            │ Auth          │     │ Image         │     │ Face /        │
            │ Controller    │     │ Controller    │     │ Recommendation│
            │               │     │               │     │ Controller    │
            └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
                    │                     │                     │
                    ▼                     ▼                     ▼
            ┌───────────────┐     ┌───────────────────────────────────────┐
            │ Auth Service  │     │ Image Validation Pipeline             │
            │ (hash,        │     │   → ImageValidationService           │
            │  session)     │     │   → valid path / error                │
            └───────┬───────┘     └───────────────┬───────────────────────┘
                    │                             │
                    ▼                             ▼
            ┌───────────────┐             ┌───────────────────────────────────────┐
            │ User, Session │             │ Analysis Pipeline                      │
            │ (models)      │             │   → FaceAnalysisService                │
            └───────┬───────┘             │      → FaceDetector → FaceAnalyzer     │
                    │                     │   → RecommendationService             │
                    │                     │      → Recommender (SpectacleCatalog)  │
                    │                     └───────────────┬───────────────────────┘
                    │                                     │
                    ▼                                     ▼
            ┌───────────────────────────────────────────────────────────────────┐
            │                         SQLite Database                            │
            │   users │ sessions │ face_analyses │ spectacles │ recommendations  │
            └───────────────────────────────────────────────────────────────────┘
                    │                                     │
                    │                     ┌───────────────┴───────────────┐
                    │                     ▼                               ▼
                    │             ┌───────────────┐               ┌───────────────┐
                    │             │ Face Detector │               │ Spectacle    │
                    │             │ Face Analyzer │               │ Catalog &    │
                    │             │ (ML)          │               │ Recommender  │
                    │             └───────────────┘               └───────────────┘
                    │                     │                               │
                    └─────────────────────┴───────────────────────────────┘
                                          │
                                    (Read/Write)
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                      ┌───────────────┐       ┌───────────────┐
                      │ Uploaded     │       │ Model artifacts│
                      │ image file   │       │ (optional)    │
                      └───────────────┘       └───────────────┘
```

**Summary flows:**

1. **Auth:** Client → Auth View → Auth Controller → Auth Service → User/Session models → SQLite.
2. **Upload:** Client → Upload View → Image Controller → Image Validation Pipeline → (valid file path or error).
3. **Analyze & Recommend:** Client → Analysis/Recommendation View → Controller → Analysis Pipeline:
   - Validation (reuse or re-run) → FaceAnalysisService (FaceDetector, FaceAnalyzer) → persist FaceAnalysis;
   - RecommendationService (Recommender + SpectacleCatalog) → persist Recommendation;
   - Response back to client.

---

## 4. Database Schema

### 4.1 Tables Overview

| Table | Purpose |
|-------|--------|
| `users` | Registered users (auth). |
| `sessions` | Active sessions or refresh tokens (auth). |
| `face_analyses` | One record per analysis run (image ref, face shape, metadata). |
| `spectacles` | Master catalog of spectacle styles (reference for recommendations). |
| `recommendations` | Recommended spectacles per face analysis (many-to-many link + score). |

### 4.2 Entity-Relationship (Text)

```
users (1) ──────────< (N) sessions
   │
   └────── (1) ──────────< (N) face_analyses
                                    │
                                    ├── (N) ──────────> (1) spectacles  [via recommendations]
                                    │
                                    └──── (1) ──────────< (N) recommendations
                                                              │
                                                              └──── (N) ──────────> (1) spectacles
```

### 4.3 Table Definitions

**users**

| Column       | Type         | Constraints / Notes                    |
|-------------|--------------|----------------------------------------|
| id          | INTEGER      | PRIMARY KEY, auto-increment            |
| email       | VARCHAR(255) | UNIQUE, NOT NULL                       |
| password_hash | VARCHAR(255) | NOT NULL                             |
| full_name   | VARCHAR(255) | nullable                               |
| is_active   | BOOLEAN      | NOT NULL, default True                  |
| created_at  | DATETIME     | NOT NULL, default UTC                  |
| updated_at  | DATETIME     | NOT NULL, default UTC                  |

**sessions**

| Column     | Type         | Constraints / Notes                    |
|------------|--------------|----------------------------------------|
| id         | INTEGER      | PRIMARY KEY, auto-increment            |
| user_id    | INTEGER      | NOT NULL, FK → users(id), ON DELETE CASCADE |
| token      | VARCHAR(255) | UNIQUE, NOT NULL (or use UUID)         |
| expires_at | DATETIME     | NOT NULL                               |
| created_at | DATETIME     | NOT NULL, default UTC                  |

**face_analyses**

| Column          | Type         | Constraints / Notes                    |
|-----------------|--------------|----------------------------------------|
| id              | INTEGER      | PRIMARY KEY, auto-increment            |
| user_id         | INTEGER      | NOT NULL, FK → users(id), ON DELETE CASCADE |
| image_path      | VARCHAR(512) | NOT NULL (or nullable if stored elsewhere) |
| face_shape      | VARCHAR(64)  | e.g. oval, round, square, heart, oblong |
| landmarks_json  | TEXT         | Optional JSON of landmark coordinates  |
| metadata_json   | TEXT         | Optional extra features / confidence   |
| status          | VARCHAR(32)  | e.g. pending, completed, failed        |
| error_message   | TEXT         | nullable, for failed runs              |
| created_at      | DATETIME     | NOT NULL, default UTC                  |

**spectacles**

| Column        | Type         | Constraints / Notes                    |
|---------------|--------------|----------------------------------------|
| id            | INTEGER      | PRIMARY KEY, auto-increment            |
| name          | VARCHAR(255) | NOT NULL                               |
| style         | VARCHAR(64)  | e.g. classic, cat-eye, aviator         |
| shape         | VARCHAR(64)  | Frame shape (for display/filter)       |
| suitable_faces| VARCHAR(255) | Comma-separated face shapes or JSON    |
| image_url     | VARCHAR(512) | nullable                               |
| description   | TEXT         | nullable                               |
| is_active     | BOOLEAN      | NOT NULL, default True                 |
| created_at    | DATETIME     | NOT NULL, default UTC                  |
| updated_at    | DATETIME     | NOT NULL, default UTC                  |

**recommendations**

| Column           | Type    | Constraints / Notes                    |
|------------------|---------|----------------------------------------|
| id               | INTEGER | PRIMARY KEY, auto-increment            |
| face_analysis_id | INTEGER | NOT NULL, FK → face_analyses(id), ON DELETE CASCADE |
| spectacle_id     | INTEGER | NOT NULL, FK → spectacles(id), ON DELETE CASCADE |
| score            | FLOAT   | NOT NULL (e.g. 0–1 or rank score)      |
| rank             | INTEGER | Optional display order (1, 2, 3…)      |
| created_at       | DATETIME| NOT NULL, default UTC                  |
| UNIQUE(face_analysis_id, spectacle_id) | | One recommendation per (analysis, spectacle) |

### 4.4 Indexes (Recommended)

- `users`: INDEX on `email`.
- `sessions`: INDEX on `token`, INDEX on `user_id`, INDEX on `expires_at`.
- `face_analyses`: INDEX on `user_id`, INDEX on `created_at`, INDEX on `status`.
- `recommendations`: INDEX on `face_analysis_id`, INDEX on `spectacle_id`.
- `spectacles`: INDEX on `style`, INDEX on `is_active`.

---

## 5. Design Notes

- **MVC:** Views = HTTP layer; Controllers = use-case orchestration; Models = persistence only. Services and pipelines encapsulate reusable and pipeline logic.
- **Auth:** Session or JWT in `sessions`; passwords only stored hashed (e.g. bcrypt). Protect all analysis/recommendation endpoints with auth.
- **Image validation:** Enforce max size and allowed MIME types; optionally max dimensions. Reject invalid files before they reach ML code.
- **Modularity:** ML components (`ml/`) can be swapped (e.g. different face detector) without changing controllers or views. Pipelines make the “validate → analyze → recommend” flow explicit and testable.
- **Production readiness:** Use `instance/` for SQLite and secrets; env-based config; migrations for schema changes; health endpoint for deployment; optional rate limiting and request logging.

This document is the single source of truth for folder structure, module roles, data flow, and database schema before implementation.
