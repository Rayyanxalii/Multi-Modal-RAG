"""Root compatibility entry point delegating to backend.server:app."""

import sys
from pathlib import Path

# Ensure project root, backend, and backend/src are available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "backend"), str(PROJECT_ROOT / "backend" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.server import (  # noqa: F401
    app,
    health_check,
    serve_home,
    get_documents,
    ChatRequest,
    resolve_asset_url,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
