# app/schemas.py
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import List, Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    selected_topics: List[str]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TopicSelection(BaseModel):
    topics: List[str]

class NewsOut(BaseModel):
    id: int
    title: str
    content: str
    category: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class InteractionCreate(BaseModel):
    action: str # 'like' or 'dislike'
