from fastapi import FastAPI
from contextlib import asynccontextmanager
from sqlalchemy import text

from app.database import engine, Base
from app.routers import auth, user, news, chat, pages

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        # Tạo bảng user_summaries nếu chưa có (Migration basic)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_summaries (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
    
    from app.ml_models import ml_models
    ml_models.load_models()
    
    yield

app = FastAPI(
    title="News Recommendation System",
    description="A modular FastAPI news system with AI chat and RAG.",
    version="2.0.0",
    lifespan=lifespan
)

# Include Routers
app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(news.router)
app.include_router(user.router)