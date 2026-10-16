#!/bin/sh
# Container entrypoint.  APP=ui (default) | api
set -e
case "${APP:-ui}" in
  ui)
    # Hugging Face Spaces embeds apps in an iframe, which needs CORS/XSRF relaxed.
    EXTRA=""
    if [ -n "$SPACE_ID" ]; then EXTRA="--server.enableCORS=false --server.enableXsrfProtection=false"; fi
    # shellcheck disable=SC2086
    exec streamlit run ui/app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 \
      --browser.gatherUsageStats=false $EXTRA ;;
  api)
    exec uvicorn copilot.service:create_app --factory --host 0.0.0.0 --port "${PORT:-8000}" ;;
  *)
    echo "unknown APP='$APP' (use ui or api)" >&2; exit 2 ;;
esac
