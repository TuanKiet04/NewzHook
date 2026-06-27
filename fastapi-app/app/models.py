# app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
from app.database import Base

class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    selected_topics = Column(ARRAY(String), default=[])
    created_at = Column(DateTime, server_default=text("now()"))

    interactions = relationship("Interaction", back_populates="user")
    chats = relationship("ChatHistory", back_populates="owner")
    summaries = relationship("UserSummary", back_populates="owner")

    cluster_id = Column(Integer, nullable=True)
    persona_prompt = Column(Text, nullable=True)
    topics = Column(ARRAY(String), default=[])
    custom_instruction = Column(Text, nullable=True)
    custom_system_prompt = Column(Text, nullable=True)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("public.users.id"))
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=text("now()"))

    owner = relationship("User", back_populates="chats")

class News(Base):

    __tablename__ = "news"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    vector = Column(Vector(768))  # Giả sử vector size 1536 (OpenAI standard)
    created_at = Column(DateTime, server_default=text("now()"))

    interactions = relationship("Interaction", back_populates="news")

class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("public.users.id"))
    news_id = Column(Integer, ForeignKey("public.news.id"))
    action = Column(String(50))  # 'like', 'dislike'
    created_at = Column(DateTime, server_default=text("now()"))

    user = relationship("User", back_populates="interactions")
    news = relationship("News", back_populates="interactions")

class UserSummary(Base):
    __tablename__ = "user_summaries"
    __table_args__ = {"schema": "public"}
 
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("public.users.id"), nullable=False)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=text("now()"))
 
    owner = relationship("User", back_populates="summaries")

class Persona(Base):
    __tablename__ = "personas"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(100))
    base_prompt = Column(Text)
    description = Column(Text)