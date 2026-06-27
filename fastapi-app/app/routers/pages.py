#app/routers/pages.py
from datetime import datetime
from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text, update
from app import models, database
from app.services.personalization import get_persona_prompts

router = APIRouter(
    tags=["Pages"]
)

templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("session_user")
    response.delete_cookie("access_token")
    return response

@router.get("/news", response_class=HTMLResponse)
async def get_news_page(request: Request, db: AsyncSession = Depends(database.get_db), page: int = 1):
    current_user = request.cookies.get("session_user")
    if not current_user:
        return RedirectResponse(url="/login")
    
    # Check if onboarding completed
    user_query = select(models.User).where(models.User.username == current_user)
    user_res = await db.execute(user_query)
    user = user_res.scalar_one_or_none()
    if not user or user.cluster_id is None:
        return RedirectResponse(url="/onboarding")
    
    size = 20 # Items per page
    offset = (page - 1) * size
    
    # Query from raw_data as per current implementation
    query = text(f"""
        SELECT 
    id,
    title,
    url,
    topic,
    published_at,
    to_char(
        published_at AT TIME ZONE 'Asia/Ho_Chi_Minh',
        'DD/MM/YYYY HH24:MI'
    ) as published_at_str,
    LEFT(content, 150) as snippet
FROM raw_data
ORDER BY published_at DESC
        LIMIT {size} OFFSET {offset}
    """)
    
    try:
        res = await db.execute(query)
        articles = [dict(row) for row in res.mappings().all()]
    except Exception as e:
        print(f"Lỗi: {e}")
        articles = [] 

    # Fetch chat turn count
    from sqlalchemy import func
    turn_count_query = select(func.count(models.ChatHistory.id)).where(models.ChatHistory.user_id == user.id)
    turn_count_res = await db.execute(turn_count_query)
    turn_count = turn_count_res.scalar() or 0

    # Query persistent chat history for UI hydration
    chat_history_query = select(models.ChatHistory).where(models.ChatHistory.user_id == user.id).order_by(models.ChatHistory.created_at.asc())
    chat_history_res = await db.execute(chat_history_query)
    chat_history = chat_history_res.scalars().all()
    
    INITIAL_PERSONA_PROMPT, REFINED_SYSTEM_PROMPT = await get_persona_prompts(user, db)
    
    if turn_count > 5:
        active_system_prompt = REFINED_SYSTEM_PROMPT
        active_mode = "REFINED_SYSTEM_PROMPT"
    else:
        active_system_prompt = INITIAL_PERSONA_PROMPT
        active_mode = "INITIAL_PERSONA_PROMPT"

    override_active = "Yes" if user.custom_instruction else "No"
    # If custom instruction is active, the overall active prompt mode matches USER_CUSTOM_INSTRUCTION style
    display_active_mode = "USER_CUSTOM_INSTRUCTION" if user.custom_instruction else active_mode

    return templates.TemplateResponse("news.html", {
        "request": request, 
        "username": current_user,
        "articles": articles,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1,
        "custom_instruction": user.custom_instruction,
        "active_system_prompt": active_system_prompt,
        "active_prompt_mode": display_active_mode,
        "turn_count": turn_count,
        "override_active": override_active,
        "chat_history": chat_history
    })

import numpy as np
from app.ml_models import ml_models

@router.get("/onboarding", response_class=HTMLResponse)
async def onboarding_page(request: Request):
    current_user = request.cookies.get("session_user")
    if not current_user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("onboarding.html", {"request": request})

@router.post("/onboarding")
async def process_onboarding(request: Request, db: AsyncSession = Depends(database.get_db)):
    current_user_name = request.cookies.get("session_user")
    if not current_user_name:
        return RedirectResponse(url="/login")
    
    data = await request.json()
    topics = data.get("topics", [])
    
    if not topics:
        return {"status": "error", "message": "No topics selected"}

    # Inference logic
    try:
        # Guarantee ML models are loaded
        if ml_models.centroids_norm is None or ml_models.topic_centroids is None:
            ml_models.load_models()

        topic_vecs = [ml_models.topic_centroids[t] for t in topics if t in ml_models.topic_centroids]
        if not topic_vecs:
            assigned_cluster = 3 # Default fallback
        else:
            # 1. compute mean vector
            user_vec = np.mean(topic_vecs, axis=0)
            # 2. normalize vector
            user_vec = user_vec / np.linalg.norm(user_vec)
            # 3. compute cosine similarity: similarities = centroids_norm @ user_vec
            similarities = ml_models.centroids_norm @ user_vec
            # 4. assign: assigned_cluster = argmax(similarities)
            assigned_cluster = int(np.argmax(similarities))
        
        # 5. Save: cluster_id, topics, selected_topics (and clear the old persona_prompt column)
        stmt = update(models.User).where(models.User.username == current_user_name).values(
            cluster_id=assigned_cluster,
            topics=topics,
            selected_topics=topics,
            persona_prompt=None
        )
        await db.execute(stmt)
        await db.commit()
        
        return {"status": "success", "cluster": assigned_cluster, "redirect": "/news"}
    except Exception as e:
        print(f"Inference/Update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommended-news")
async def get_recommended_news(request: Request, db: AsyncSession = Depends(database.get_db)):
    current_user_name = request.cookies.get("session_user")
    if not current_user_name:
        return []
    
    # Get user
    query = select(models.User).where(models.User.username == current_user_name)
    res = await db.execute(query)
    user = res.scalar_one_or_none()
    
    if not user:
        return []

    articles = []
    
    # 1. Try Cluster-aware personalization first
    if user.cluster_id is not None:
        try:
            cluster_query = text("""
                SELECT 
    id,
    title,
    url,
    topic,
    published_at,
    to_char(
        published_at AT TIME ZONE 'Asia/Ho_Chi_Minh',
        'DD/MM/YYYY HH24:MI'
    ) as published_at_str,
    LEFT(content, 150) as snippet
FROM raw_data
WHERE cluster_id = :cluster_id
ORDER BY published_at DESC
LIMIT 50
            """)
            res = await db.execute(cluster_query, {"cluster_id": user.cluster_id})
            articles = [dict(row) for row in res.mappings().all()]
            print(f"Cluster-based recommendation: found {len(articles)} articles for cluster {user.cluster_id}")
        except Exception as e:
            print(f"Cluster-based recommendation error: {e}")
            articles = []

    # 2. Fallback to topic-based balanced recommendation if no articles found or cluster_id is None
    if not articles and user.topics:
        print("Fallback to topic-based recommendation...")
        try:
            for topic in user.topics:
                rec_query = text("""
                    SELECT id, title, url, topic, published_at as raw_pub,
                           to_char(published_at, 'DD/MM/YYYY HH24:MI') as published_at, 
                           LEFT(content, 150) as snippet
                    FROM raw_data
                    WHERE topic = :topic
                    ORDER BY raw_pub DESC
                    LIMIT 50
                """)
                res = await db.execute(rec_query, {"topic": topic})
                articles.extend([dict(row) for row in res.mappings().all()])
            
            # Sort globally by publication date desc
            articles = sorted(articles, key=lambda x: x['raw_pub'] if x.get('raw_pub') else datetime.min, reverse=True)
            articles = articles[:10]
        except Exception as e:
            print(f"Fallback recommendation error: {e}")
            articles = []

    return articles

from pydantic import BaseModel

class CustomInstructionRequest(BaseModel):
    instruction: str

@router.post("/save-custom-instruction")
async def save_custom_instruction(request: CustomInstructionRequest, req: Request, db: AsyncSession = Depends(database.get_db)):
    current_user_name = req.cookies.get("session_user")
    if not current_user_name:
        return {"status": "error", "message": "Not authenticated"}
    
    query = select(models.User).where(models.User.username == current_user_name)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        return {"status": "error", "message": "User not found"}
        
    user.custom_instruction = request.instruction.strip()
    await db.commit()
    return {"status": "success", "message": "Custom prompt saved successfully"}

@router.post("/clear-custom-instruction")
async def clear_custom_instruction(req: Request, db: AsyncSession = Depends(database.get_db)):
    current_user_name = req.cookies.get("session_user")
    if not current_user_name:
        return {"status": "error", "message": "Not authenticated"}
    
    query = select(models.User).where(models.User.username == current_user_name)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        return {"status": "error", "message": "User not found"}
        
    user.custom_instruction = None
    await db.commit()
    return {"status": "success", "message": "Custom prompt cleared successfully"}
