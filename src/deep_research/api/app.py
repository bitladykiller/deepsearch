"""Deep Research FastAPI 应用工厂。"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import APISettings
from .routers import health_router, research_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger("deep_research").setLevel(logging.INFO)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    settings = APISettings.from_env()
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(research_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = APISettings.from_env()
    uvicorn.run(
        "deep_research.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
    )
