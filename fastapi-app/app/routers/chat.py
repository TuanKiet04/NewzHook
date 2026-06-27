#app/routers/chat.py
from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
import httpx
from app import models, database
from app.services.rag import get_rag_context, get_history_recommendations, build_system_prompt, get_rag_context_with_observability
from app.services.personalization import get_active_persona, refine_persona_prompt, get_persona_prompts

router = APIRouter(
    tags=["Chat"]
)

VLLM_URL = "http://10.6.21.3:8888"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

def is_conversational_query(query: str) -> bool:
    """
    STRICT mode: Chỉ return True cho những câu CHẮC CHẮN là conversational.
    Tất cả câu hỏi khác → dùng RAG (return False)
    
    Mục tiêu: Minimize general knowledge, maximize RAG usage
    """
    q = query.lower().strip()
    
    # ===== CASE 1: Very short greetings (≤ 2 words) =====
    if len(q.split()) <= 2:
        short_greetings = ["chào", "hi", "hello", "hey", "alo", "lô"]
        if any(g in q for g in short_greetings):
            return True
    
    # ===== CASE 2: Pure identity & small talk queries =====
    # These are GENUINELY conversational, not knowledge-seeking
    pure_conversational = [
        "bạn là ai",
        "who are you",
        "tên bạn",
        "tên bạn là gì",
        "bạn tên gì",
        "tên của bạn",
        "giới thiệu bản thân",
        "tự giới thiệu",
        "bạn là gì",
        "you are",
        "cảm ơn",
        "thank you",
        "thanks",
        "bye",
        "tạm biệt",
        "goodbye",
        "see you"
    ]
    
    if any(p in q for p in pure_conversational):
        return True
    
    return False

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat_with_ai(request: ChatRequest, req: Request, db: AsyncSession = Depends(database.get_db)):
    current_user_name = req.cookies.get("session_user")
    if not current_user_name:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    # 1. Fetch User Profile
    query = select(models.User).where(models.User.username == current_user_name)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # 2. Get Active Persona Prompt & Mode dynamically based on chat turn count
    # Fetch historical chat turn count before this turn starts
    from sqlalchemy import func
    turn_count_query = select(func.count(models.ChatHistory.id)).where(models.ChatHistory.user_id == user.id)
    turn_count_res = await db.execute(turn_count_query)
    history_count = turn_count_res.scalar() or 0

    INITIAL_PERSONA_PROMPT, REFINED_SYSTEM_PROMPT = await get_persona_prompts(user, db)

    # Priority mapping: USER_CUSTOM_INSTRUCTION (if custom_instruction is set) > REFINED_SYSTEM_PROMPT (if history_count >= 5) > INITIAL_PERSONA_PROMPT
    if user.custom_instruction:
        active_mode = "USER_CUSTOM_INSTRUCTION"
        user_persona = user.custom_instruction
    elif history_count >= 5:
        active_mode = "REFINED_SYSTEM_PROMPT"
        user_persona = REFINED_SYSTEM_PROMPT
    else:
        active_mode = "INITIAL_PERSONA_PROMPT"
        user_persona = INITIAL_PERSONA_PROMPT
    
    # 3. Detect intent & Retrieve RAG Context optionally
    import time
    is_general = is_conversational_query(request.message)
    
    context = ""
    top_chunks = []
    embed_latency = 0.0
    retrieval_latency = 0.0
    log_source = "AI Assistant (General knowledge)"
    
    if is_general:
        print(f"[HYBRID RAG] Conversational query detected ('{request.message}'). Skipping database retrieval.")
    else:
        # Run retrieval and track observability latencies
        context, top_chunks, embed_latency, retrieval_latency = await get_rag_context_with_observability(request.message, db)
        
        # Verify confidence threshold: if max similarity is below 0.70, fallback to general conversational knowledge
        max_similarity = max([c["similarity"] for c in top_chunks]) if top_chunks else 0.0
        if max_similarity < 0.60:
            print(f"[HYBRID RAG] Retrieval confidence low ({max_similarity:.4f} < 0.60). Falling back to general AI assistant mode.")
            context = ""
            top_chunks = []
            log_source = "AI Assistant (No matching articles found)"
        else:
            top_chunk = top_chunks[0]
            log_source = f'<a href="{top_chunk["url"]}" target="_blank" style="color: #ffcc00; text-decoration: underline; font-weight: bold;">{top_chunk["title"]}</a>'
            
    # 4. Build System Prompt with Grounding Rules & Active Persona
    system_prompt = build_system_prompt(context, user_persona)
    
    # Grounding rules are ONLY attached when retrieved documents are actually used!
    if context:
        system_prompt += "\n=== GROUNDING RULES ===\n- ALWAYS prioritize facts from the provided context.\n- If the context doesn't contain the answer, say honestly that you don't know based on the provided context."
        
    # 5. Call LLM for Assistant Response and measure generation latency
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.message}
    ]
    
    reply = None
    gen_start = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            vllm_response = await client.post(
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": VLLM_MODEL,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.4
                }
            )
            vllm_response.raise_for_status()
            reply = vllm_response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"vLLM Chat error: {e}")
        reply = "Xin lỗi, tôi gặp sự cố kết nối với mô hình ngôn ngữ."
        
    gen_latency = time.time() - gen_start
    
    print(f"\n[DEMO LOG] Live RAG + vLLM Generation completed. Mode: {active_mode} | Source: {log_source}\n")

    # 6. Save Turn to ChatHistory Table (persists history needed for future refinement)
    new_chat = models.ChatHistory(
        user_id=user.id,
        message=request.message,
        response=reply
    )
    db.add(new_chat)
    await db.commit()
    await db.refresh(new_chat)
    
    # 7. Fetch all historical chat turns for personalization & recommendations
    history_query = select(models.ChatHistory).where(models.ChatHistory.user_id == user.id).order_by(models.ChatHistory.created_at.asc())
    history_res = await db.execute(history_query)
    history_rows = history_res.scalars().all()
    history_count = len(history_rows)
    
    # 8. Dynamic Refinement (Prepared architecture for future refinement)
    # Lịch sử hội thoại đã được lưu trữ hoàn toàn trong bảng chat_history để phục vụ việc tinh chỉnh prompt tự động trong tương lai.
    # Flow modular/extensible được giữ nguyên trong app/services/personalization.py.
    if history_count > 0 and history_count % 5 == 0 and not user.custom_instruction:
        print(f"[CƠ CHẾ TINH CHỈNH PROMPT] Ngưỡng {history_count} turns đã đạt. Lịch sử hội thoại đã được chuẩn bị sẵn sàng cho việc tự động tinh chỉnh prompt của '{user.username}'.")
        
    # 9. Popup Recommendation System (Triggers every 5 turns)
    recommendations = []
    show_recommendation_popup = False
    if history_count > 0 and history_count % 5 == 0:
        show_recommendation_popup = True
        try:
            # Build semantic search vector query using the last few user chat messages
            recent_turns = [h.message for h in history_rows[-5:] if h.message]
            history_text = " ".join(recent_turns)
            recommendations = await get_history_recommendations(history_text, db, limit=3)
        except Exception as e:
            print(f"Error querying semantic history recommendations: {e}")
            recommendations = []
        
    import statistics
    similarities = [c["similarity"] for c in top_chunks]
    avg_similarity = statistics.mean(similarities) if similarities else 0.0

    metrics_payload = {
        "embed_latency": embed_latency,
        "retrieval_latency": retrieval_latency,
        "generation_latency": gen_latency,
        "avg_similarity": avg_similarity,
        "top_chunks": top_chunks
    }

    updated_system_prompt = REFINED_SYSTEM_PROMPT if history_count > 5 else INITIAL_PERSONA_PROMPT
    override_active = "Yes" if user.custom_instruction else "No"
    display_active_mode = "USER_CUSTOM_INSTRUCTION" if user.custom_instruction else ("REFINED_SYSTEM_PROMPT" if history_count > 5 else "INITIAL_PERSONA_PROMPT")

    return {
        "response": reply,
        "recommendations": recommendations,
        "show_recommendation_popup": show_recommendation_popup,
        "active_prompt_mode": display_active_mode,
        "log_source": log_source,
        "matched_qa_id": None,
        "similarity_score": avg_similarity,
        "turn_count": history_count,
        "active_system_prompt": updated_system_prompt,
        "override_active": override_active,
        "metrics": metrics_payload
    }