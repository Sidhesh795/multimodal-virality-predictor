"""Launch the Virality dashboard API locally or on a hosted service."""

import os
import uvicorn

if __name__ == "__main__":
    # Reload mode starts a Windows subprocess and is unnecessary for the
    # presentation build. A single stable process also gives cleaner startup
    # behavior for local and hosted deployments.
    uvicorn.run(
        "backend.app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
