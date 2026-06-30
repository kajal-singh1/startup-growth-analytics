# Deployment Guide — Module 13

Both services below are **completely free** and deploy directly from your
GitHub repository. No credit card required for either.

---

## Part A — Deploy the Dashboard (Streamlit Community Cloud)

### Step 1: Verify locally first
```bash
python scripts\verify_deployment.py
```
Confirm it shows `0 failed` before continuing.

### Step 2: Commit and push everything
```bash
git add .
git commit -m "Module 13: Deployment configs — Streamlit Cloud + Render ready"
git push
```

### Step 3: Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository**: `your-username/startup-growth-analytics`
   - **Branch**: `main`
   - **Main file path**: `scripts/dashboard.py`
5. Click **"Advanced settings"** and set:
   - **Python version**: `3.12`
6. Before clicking Deploy, you need Streamlit Cloud to use your lean
   requirements file instead of the full `requirements.txt`. Two options:

   **Option A (recommended) — rename for deployment:**
   ```bash
   git mv requirements.txt requirements_full.txt
   git mv requirements_dashboard.txt requirements.txt
   git commit -m "Use lean requirements for Streamlit Cloud"
   git push
   ```
   Streamlit Cloud always reads `requirements.txt` by default — this swap
   makes it install only Streamlit/Pandas/Plotly (fast, fits free tier RAM).

   **Option B — keep both, accept slower build:**
   Leave as-is; Streamlit Cloud will install everything in your full
   `requirements.txt` including TensorFlow, which may exceed the 1GB RAM
   limit on the free tier and fail. Option A is strongly recommended.

7. Click **Deploy**. First build takes 3-5 minutes.
8. Your dashboard will be live at:
   `https://your-username-startup-growth-analytics.streamlit.app`

### Step 4: Auto-deploy on every push
No action needed — Streamlit Cloud watches your GitHub repo and
redeploys automatically every time you `git push` to `main`.

---

## Part B — Deploy the API (Render)

### Step 1: Go to Render
1. Visit **https://render.com**
2. Sign in with GitHub
3. Click **"New +"** → **"Blueprint"**

### Step 2: Connect your repo
1. Select `your-username/startup-growth-analytics`
2. Render will detect `render.yaml` automatically and show:
   - Service: `startup-growth-analytics-api`
   - Plan: Free
   - Build: `pip install -r requirements_api.txt`
   - Start: `uvicorn scripts.api:app --host 0.0.0.0 --port $PORT`
3. Click **"Apply"**

### Step 3: Wait for build
First deploy takes 3-5 minutes. Watch the live logs in the Render dashboard.

### Step 4: Get your live API URL
Once deployed, Render gives you a URL like:
```
https://startup-growth-analytics-api.onrender.com
```
Visit `https://startup-growth-analytics-api.onrender.com/docs` to see
your live Swagger UI — identical to what you saw locally.

### Important — Free tier sleep behavior
Render's free tier **spins down after 15 minutes of inactivity** and
takes ~30-60 seconds to wake up on the next request. This is normal
and expected for the free tier; mention it if including the API link
in your portfolio/resume so reviewers aren't confused by the first
slow request.

---

## Part C — Final Verification Checklist

After both are deployed, confirm:

- [ ] Dashboard loads at your `.streamlit.app` URL
- [ ] All 8 dashboard pages render without errors
- [ ] Country Analysis page shows charts for at least 3 countries
- [ ] API `/docs` loads at your `.onrender.com` URL
- [ ] API `/` health check returns `"status": "ok"`
- [ ] API `/countries` returns 15 countries
- [ ] API `/predict` returns a prediction (heuristic or model-based)

---

## Part D — Add Links to Your README

Open `README.md` and add near the top:

```markdown
## 🌐 Live Demo

- **Dashboard**: https://your-username-startup-growth-analytics.streamlit.app
- **API Docs**: https://startup-growth-analytics-api.onrender.com/docs
```

Commit:
```bash
git add README.md
git commit -m "Add live deployment links to README"
git push
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Streamlit Cloud build fails on TensorFlow | You skipped the requirements.txt swap in Part A Step 6 — do it now |
| Dashboard shows "No data found" | CSVs were gitignored — re-check `.gitignore` doesn't block `data/processed/` |
| Render build fails on scikit-learn version | Check `requirements_api.txt` pins a version compatible with Python 3.12 |
| API `/predict` always uses heuristic | `models/saved/best_model.pkl` wasn't committed — check it's not gitignored |
| Render app sleeps and first request times out | Normal free-tier behavior — wait 30-60s and retry |

---

## What You've Built

A fully deployed, end-to-end data science system:

```
GitHub Repo (source of truth)
       │
       ├──→ Streamlit Cloud ──→ Live Dashboard (8 pages)
       │
       └──→ Render ──→ Live API (11 endpoints)
```

Both auto-deploy on every `git push`. This is a genuine production
deployment pattern used by real companies — not a toy demo.
