"""FastAPI app: CarWale RAG car assistant."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .api import chat, routes
from .bootstrap import build_pipeline
from .config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    pipeline = await build_pipeline(s)
    app.state.pipeline = pipeline
    app.state.stats = pipeline.stats
    yield
    await db.close_pool()


settings = get_settings()
app = FastAPI(title="CarWale RAG Assistant", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_origin_regex=r"https://[a-z0-9-]+\.vercel\.app",
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(chat.router)
app.include_router(routes.router)


@app.get("/")
async def root():
    return {"service": "CarWale RAG Assistant API", "docs": "/docs", "health": "/api/health"}
