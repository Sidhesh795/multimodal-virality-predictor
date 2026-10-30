"""
FastAPI Backend Application Entrypoint for Multimodal Social Media Virality Prediction.
Configures FastAPI instance, CORS middleware, lifespan events, and API routers.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import BASE_DIR
from backend.app.routes.predict_routes import router as predict_router, get_predictor, get_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to pre-warm models and extractors on application cold start.
    """
    print("Prewarming Virality Predictor...")
    get_predictor()
    # The feature pipeline includes Sentence-Transformer initialization, which
    # is intentionally lazy-loaded on the first prediction request. This keeps
    # health checks, deployment probes, and API startup responsive.
    print("Application prewarming complete. Feature pipeline will load on demand.")
    yield
    print("Shutting down API service.")


def create_app() -> FastAPI:
    """Factory function to build and configure the FastAPI app instance."""
    app = FastAPI(
        title="Multimodal Virality Prediction API",
        description="AI-powered API for predicting social media virality, SHAP attributions, and recommendations.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS Middleware configuration for Phase 6 frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API v1 routes
    app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])

    # Serve dashboard last so /api/v1 routes take precedence
    app.mount(
        "/",
        StaticFiles(directory=str(BASE_DIR / "frontend"), html=True),
        name="frontend",
    )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
