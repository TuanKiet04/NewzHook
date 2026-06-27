# app/database.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Thay đổi postgresql:// thành postgresql+asyncpg:// để sử dụng asyncpg
DATABASE_URL = 'postgresql+asyncpg://kietcorn:kiietqo9204@10.6.21.3:5432/optimize'

engine = create_async_engine(DATABASE_URL, echo=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()

# Dependency để sử dụng trong các API (Async version)
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()