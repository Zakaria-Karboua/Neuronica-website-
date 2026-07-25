# Neuronica

**Live demo:** https://neuronica.onrender.com *(free-tier hosting — first load after
inactivity can take 30-60s to wake up)*

<!--
  TODO: add 3-4 screenshots here before sharing this repo as a portfolio piece —
  this is usually the first thing a recruiter/reviewer looks at before reading
  setup instructions. Suggested shots: home page hero, a lesson page (showing
  syntax highlighting), the profile/trophy page, and Mission Control (Pomodoro timer).

  ![Home page](docs/screenshots/home.png)
  ![Lesson page](docs/screenshots/lesson.png)
  ![Profile & trophies](docs/screenshots/profile.png)
  ![Mission Control](docs/screenshots/mission-control.png)
-->

A space-themed learning platform: curriculum content (`.md` lessons + `.ipynb` projects)
grouped into phases ("star systems"), with Google/GitHub login and Pomodoro-style
focus-time tracking ("Mission Control").

## 1. Local setup

```bash
cd neuronica
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:
- `DJANGO_SECRET_KEY` — any long random string (e.g. `python3 -c "import secrets; print(secrets.token_urlsafe(50))"`)
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` — from your Google Cloud Console OAuth client
- `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` — from your GitHub OAuth App

`.env` is git-ignored — never commit it.

## 3. Database

```bash
python3 manage.py migrate
python3 manage.py createsuperuser   # so you can access /admin/
```

## 4. Import your content

Each phase is a folder of numbered `.md` files (`01-topic.md`, `02-topic.md`, ...)
and optionally `.ipynb` files:

```bash
python3 manage.py import_phase /path/to/phase1_extracted --number 1 --title "Programming Foundations"
```

Run this again any time you add or edit files in that folder — it updates existing
lessons/projects in place (matched by slug), it doesn't duplicate them.

Preview what a run would do without touching the database:
```bash
python3 manage.py import_phase /path/to/phase1_extracted --number 1 --title "..." --dry-run
```

Adding **Phase 2** later is exactly the same command with `--number 2` and its own folder.
No code changes needed.

If `jupyter nbconvert` isn't installed yet, add `--skip-notebooks` to import just the lessons.

## 5. Run it

```bash
python3 manage.py runserver
```

Visit `http://localhost:8000`.

## 6. Wiring OAuth login

The Google/GitHub OAuth apps you already registered point at:
- `http://localhost:8000/accounts/google/login/callback/`
- `http://localhost:8000/accounts/github/login/callback/`

Once your `.env` has real client IDs/secrets, the "Login with Google" / "Login with GitHub"
links in the nav bar will work immediately — no extra Django config needed.

## 7. AI Tutor (Gemini-backed chat)

Neuronica includes a right-docked AI tutor (the "🤖 AI Tutor" tab, visible once logged in)
that answers questions about the curriculum. It's powered by Google's Gemini API, on the
**free tier** — no cost, but with real rate limits worth understanding before you rely on it.

**What it does:**
- If you're on a lesson page when you ask a question, that lesson's markdown content is
  automatically injected into the AI's system prompt — so "explain this part" actually
  works instead of answering blind. See `tutor/views.py::_build_system_instruction`.
- Conversation history persists per user in the `ChatMessage` model (`tutor/models.py`),
  so reopening the panel later shows past questions. The last 10 turns are sent back to
  Gemini as context on every new question (`HISTORY_TURNS` in `tutor/views.py`).
- Failure modes are handled gracefully rather than crashing: missing API key → clear
  "not configured yet" message; Gemini rate-limited (HTTP 429) → "try again in a minute";
  any other API error → generic retry message. See `tutor/views.py::ask`.

**Setup:**
1. Get a free key at `https://aistudio.google.com/apikey` (no credit card needed)
2. Add to `.env`:
   ```
   GEMINI_API_KEY=your-key-here
   GEMINI_MODEL=gemini-3.1-flash-lite
   ```
3. That's it — the tutor tab works immediately for any logged-in user

**Rate limits — read this before assuming it'll always work:**
Gemini's free tier is a quota **per API key, shared across every user of your site** —
not a per-user allowance. With `gemini-3.1-flash-lite` you get roughly 1,000–1,500
requests/day total, shared by everyone using Neuronica. If that gets exhausted, users
see the "try again in a minute" message rather than a broken feature — but it does mean
a busy day could throttle everyone. If Neuronica grows a real audience, revisit this
(upgrade to a paid Gemini tier, or split traffic across multiple free-tier keys).

**Model name churn:** Google's Gemini model lineup changes fairly often — model IDs get
deprecated with only a few months' notice (this happened once already during development:
`gemini-2.5-flash-lite` was deprecated the same week this project was being built). If the
tutor suddenly starts returning "AI tutor error (404)", the model name in `GEMINI_MODEL`
is almost always the cause — check Google's current model list and update the env var.

**API endpoints** (`tutor/urls.py`):
- `POST /tutor/ask/` — send a question, get a reply (also saves both to history)
- `GET /tutor/history/` — load recent messages, optionally filtered by `?lesson_id=`

## Project structure

```
neuronica/
├── curriculum/        # Phase, Lesson, Project models + import_phase command
├── focus/              # Pomodoro sessions, streaks, dashboard
├── accounts/           # (reserved for any custom profile fields later)
├── templates/          # base.html (theme+starfield+nav) + per-app templates
├── static/css/theme.css   # light/dark space palette (CSS custom properties)
├── static/js/          # starfield.js, theme.js, timer.js
└── neuronica_config/   # settings.py, urls.py
```

## Deploying for free — Neon (database) + Render (hosting)

Render's own free Postgres database gets **permanently deleted after 30 days** — not something
to build on. Instead, use **Neon** for the database (a genuinely permanent free tier, ~0.5GB,
never expires) paired with **Render** for hosting the app itself.

### 1. Push this project to GitHub

```bash
cd neuronica
git init
git add .
git commit -m "Initial Neuronica deploy"
git remote add origin https://github.com/<your-username>/neuronica.git
git push -u origin main
```

### 2. Create your free Neon database

1. Go to `https://neon.tech` → sign up (no credit card needed) → "Create a project"
2. Once created, copy the **connection string** shown (starts with `postgresql://...`)
   — you'll paste this into Render as `DATABASE_URL` in step 4.

### 3. Deploy to Render using the included Blueprint

1. Go to `https://render.com` → sign up → **New → Blueprint**
2. Connect your GitHub account and select the `neuronica` repo
3. Render reads `render.yaml` automatically and shows you the `neuronica` web service —
   click **Apply**

### 4. Set the environment variables Render couldn't fill in automatically

In the Render dashboard, under your service → **Environment**, add:

| Key | Value |
|---|---|
| `DATABASE_URL` | your Neon connection string from step 2 |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | from Google Cloud Console |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | from your GitHub OAuth App |
| `GEMINI_API_KEY` | from `https://aistudio.google.com/apikey` |
| `DJANGO_SUPERUSER_USERNAME` / `_EMAIL` / `_PASSWORD` | your admin login for `/admin/` |

`DJANGO_SECRET_KEY` is auto-generated by the Blueprint. `DJANGO_ALLOWED_HOSTS` and
`CSRF_TRUSTED_ORIGINS` are pre-filled for `neuronica.onrender.com` — if Render gives you
a different subdomain, update both to match.

### 5. Deploy

Render builds automatically once the env vars are saved. The build step runs
`migrate` and `bootstrap_production` for you — meaning your superuser and Phase 1
content are created automatically on first deploy, with **no shell access needed**
(Render's free tier doesn't provide one).

### 6. Update your OAuth apps with the production URL

Once you have your live URL (e.g. `https://neuronica.onrender.com`):
- **Google Cloud Console**: add `https://neuronica.onrender.com` as an authorized origin,
  and `https://neuronica.onrender.com/accounts/google/login/callback/` as a redirect URI
  (alongside your existing localhost ones — Google allows multiple)
- **GitHub OAuth App**: since GitHub only allows one callback URL per app, either edit
  your existing app to the production URL (breaks local login unless you swap back), or
  register a **second GitHub OAuth App** just for production — the cleaner option

### 7. Adding Phase 2 (and beyond) after deployment

Commit a new folder following the same naming convention:
```
content/phase2-statistics-and-probability/
    01-descriptive-statistics.md
    02-probability-theory.md
    ...
```
Push to GitHub → Render redeploys automatically → `bootstrap_production` picks up the
new phase and imports it. No code changes, no manual server access needed.

### Free-tier limitations to know

- **Render free web services spin down after 15 minutes of inactivity** — the first
  visit after a quiet period takes ~30-60s to wake up. Fine for a learning site.
- **Neon free tier scales to zero too** — similar cold-start behavior, but very fast (~500ms).
- **Neon free tier**: 0.5GB storage, 100 compute-hours/month — generous for a small
  user base, but worth monitoring as Neuronica grows.
