# Deploying the UI on Hugging Face Spaces (Docker)

1. Create a **Docker** Space (free CPU hardware is enough: all heavy reasoning is via the LLM API;
   bge-small embeddings run fine on CPU).
2. Push this repository to the Space. The Space's root `README.md` must start with this header
   (keep your normal README below it):
   ```yaml
   ---
   title: Research Copilot
   emoji: 📚
   sdk: docker
   app_port: 7860
   ---
   ```
3. In **Settings -> Variables and secrets** add:
   - `ANTHROPIC_API_KEY` (secret)
   - `COPILOT_UI_PASSWORD` (secret) - **strongly recommended**: a public Space without it lets
     anyone spend your API credit
   - `COPILOT_UI_MAX_PAPERS=15` and optionally `COPILOT_MAX_TOTAL_TOKENS=250000` (per-run token cap)
   - `COPILOT_USER_AGENT=research-copilot/1.0 (mailto:you@yourdomain)` (arXiv / Semantic Scholar etiquette)
4. The default `APP=ui` entrypoint listens on `$PORT` (7860). `scripts/start.sh` relaxes Streamlit's
   CORS/XSRF settings automatically when it detects a Space (`SPACE_ID`).

Notes
- Space storage is ephemeral: the PDF/search cache is rebuilt after a restart (just slower).
- First run after a cold start downloads nothing (the embedding model is baked into the image).
- Prefer `WITH_BGE=0` + `COPILOT_EMBEDDER=hash` only for smoke tests: hash embeddings are not semantic.
