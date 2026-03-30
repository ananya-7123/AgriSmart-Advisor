# Deployment Guide: Render (Backend) + Vercel (Frontend)

This guide deploys the complete prototype from the final-prototype branch.

## 1. Pre-Deployment Checklist

1. Push all required code and model files to GitHub on final-prototype.
2. Confirm backend model files exist at paths used by backend/app.py.
3. Confirm frontend build works locally.
4. Keep secrets in platform environment variables, not in tracked .env files.

## 2. Deploy Backend to Render

Create a new Render Web Service.

- Repository: your GitHub repo
- Branch: final-prototype
- Root Directory: backend
- Runtime: Python 3
- Build Command: pip install -r requirements.txt
- Start Command: gunicorn -w 1 -b 0.0.0.0:$PORT app:app --timeout 60

Set environment variables:

- FLASK_ENV=production
- FRONTEND_ORIGINS=https://your-vercel-project.vercel.app

If you want preview URLs to work too, add comma-separated origins:

- FRONTEND_ORIGINS=https://your-vercel-project.vercel.app,https://your-vercel-project-git-final-prototype-your-team.vercel.app

After deploy completes, verify health:

- GET https://your-render-service.onrender.com/health

Expected: HTTP 200 with status running.

## 3. Deploy Frontend to Vercel

Create a new Vercel project and import the same repository.

- Branch: final-prototype
- Framework Preset: Vite
- Root Directory: frontend
- Install Command: npm install
- Build Command: npm run build
- Output Directory: dist

Set environment variables in Vercel Project Settings:

- VITE_API_BASE_URL=https://your-render-service.onrender.com
- VITE_SUPABASE_URL=<your supabase url>
- VITE_SUPABASE_ANON_KEY=<your supabase anon key>

Deploy and open:

- https://your-vercel-project.vercel.app

## 4. Integration Validation

1. Open Home page and run ML-only prediction.
2. Run NLP-only prediction.
3. Upload a leaf image and run CNN prediction.
4. Run full assessment and verify ARI response.
5. Confirm backend logs show successful requests.

## 5. Troubleshooting

- 502/503 on backend startup:
  - Model files may be missing in deployment branch.
  - Check Render logs for FileNotFoundError.
- CORS errors in browser:
  - FRONTEND_ORIGINS does not include your exact Vercel URL.
- Frontend calls localhost in production:
  - VITE_API_BASE_URL is missing or incorrect in Vercel env settings.
- Slow first response:
  - Cold start/model loading is expected on free tier.

## 6. Post-Deployment Hardening

1. Remove tracked secrets from repository history if any were committed earlier.
2. Rotate exposed keys (Supabase anon key if previously public in repo).
3. Keep FRONTEND_ORIGINS restricted to known frontend domains.
4. Add deployment links and status badge in root README.
