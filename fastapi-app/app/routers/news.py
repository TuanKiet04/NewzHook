# app/routers/news.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List
from app import models, schemas, database, oauth2

router = APIRouter(
    prefix="/news",
    tags=["News"]
)

@router.post("/select-topics", response_model=schemas.UserOut)
async def select_topics(
    selection: schemas.TopicSelection, 
    db: AsyncSession = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):
    current_user.selected_topics = selection.topics
    await db.commit()
    await db.refresh(current_user)
    return current_user

@router.post("/{id}/like", status_code=status.HTTP_201_CREATED)
async def like_news(
    id: int, 
    db: AsyncSession = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):
    # Check if news exists
    news_query = select(models.News).where(models.News.id == id)
    news_result = await db.execute(news_query)
    if not news_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="News not found")
        
    # Check if interaction already exists
    inter_query = select(models.Interaction).where(
        models.Interaction.user_id == current_user.id,
        models.Interaction.news_id == id
    )
    inter_result = await db.execute(inter_query)
    interaction = inter_result.scalar_one_or_none()
    
    if interaction:
        interaction.action = "like"
    else:
        new_interaction = models.Interaction(user_id=current_user.id, news_id=id, action="like")
        db.add(new_interaction)
        
    await db.commit()
    return {"message": "News liked successfully"}

@router.post("/{id}/dislike", status_code=status.HTTP_201_CREATED)
async def dislike_news(
    id: int, 
    db: AsyncSession = Depends(database.get_db),
    current_user: models.User = Depends(oauth2.get_current_user)
):
    # Check if news exists
    news_query = select(models.News).where(models.News.id == id)
    news_result = await db.execute(news_query)
    if not news_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="News not found")
        
    # Check if interaction already exists
    inter_query = select(models.Interaction).where(
        models.Interaction.user_id == current_user.id,
        models.Interaction.news_id == id
    )
    inter_result = await db.execute(inter_query)
    interaction = inter_result.scalar_one_or_none()
    
    if interaction:
        interaction.action = "dislike"
    else:
        new_interaction = models.Interaction(user_id=current_user.id, news_id=id, action="dislike")
        db.add(new_interaction)
        
    await db.commit()
    return {"message": "News disliked successfully"}

@router.post("/log-interaction/{article_id}")
async def log_interaction(
    article_id: int,
    request: Request,
    db: AsyncSession = Depends(database.get_db)
):
    username = request.cookies.get("session_user")
    user = None
    
    if username:
        query = select(models.User).where(models.User.username == username)
        result = await db.execute(query)
        user = result.scalar_one_or_none()
    
    if not user:
        try:
            user = await oauth2.get_current_user(request, db)
        except:
            pass

    if user:
        db.add(models.Interaction(user_id=user.id, news_id=article_id, action="click"))
        await db.commit()
        return {"status": "success", "article_id": article_id}

    return {"status": "user_not_found"}
