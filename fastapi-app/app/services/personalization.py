# app/services/personalization.py
import httpx
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from app import models

VLLM_URL = "http://10.6.21.3:8888"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

async def chat_vllm(messages: list[dict], system: str = None) -> str:
    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)
    
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": VLLM_MODEL,
                "messages": payload_messages,
                "max_tokens": 1024,
                "temperature": 0.4,
            }
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

async def get_persona_prompts(user: models.User, db: AsyncSession) -> tuple[str, str]:
    """
    Dynamically loads or constructs the initial and refined prompts for a user
    based on their assigned cluster ID (from the public.personas table).
    Returns (INITIAL_PERSONA_PROMPT, REFINED_SYSTEM_PROMPT).
    """
    from sqlalchemy.future import select
    from app import models

    INITIAL_PERSONA_PROMPT = None
    persona_obj = None

    if user.cluster_id is not None:
        p_query = select(models.Persona).where(models.Persona.cluster_id == user.cluster_id)
        p_result = await db.execute(p_query)
        persona_obj = p_result.scalar_one_or_none()
        if persona_obj:
            INITIAL_PERSONA_PROMPT = persona_obj.base_prompt

    if not INITIAL_PERSONA_PROMPT:
        # Fallback to general persona if cluster is empty/missing
        INITIAL_PERSONA_PROMPT = """Bạn là trợ lý tin tức đa năng, cởi mở và linh hoạt. Người dùng bạn đang phục vụ có sở thích đa dạng, không giới hạn ở một lĩnh vực cụ thể. Hãy giao tiếp theo phong cách tự nhiên, thân thiện và dễ gần."""

    # Build dynamically refined system prompt by merging the stable base persona
    # with the soft, evolved conversational preferences and interests of the user
    REFINED_SYSTEM_PROMPT = INITIAL_PERSONA_PROMPT
    if user.persona_prompt:
        REFINED_SYSTEM_PROMPT += f"\n\n=== PHONG CÁCH VÀ SỞ THÍCH HỘI THOẠI ĐÃ ĐÚC KẾT CỦA NGƯỜI DÙNG ===\n- {user.persona_prompt}\n(Hãy linh hoạt điều chỉnh giọng điệu, đáp ứng các định dạng yêu thích và phản ánh các chủ đề quan tâm trên trong câu trả lời của bạn một cách tự nhiên.)"
    else:
        # Fallback recommendations if no history has been refined yet
        cluster_name = persona_obj.name if persona_obj else "tin tức chuyên sâu"
        cluster_desc = persona_obj.description if persona_obj else "Giao tiếp tự nhiên, thân thiện và linh hoạt."
        REFINED_SYSTEM_PROMPT += f"\n\n=== KHUYẾN NGHỊ PHONG CÁCH CỦA PHÂN KHÚC ===\n- Người dùng thuộc nhóm quan tâm đến {cluster_name}: {cluster_desc}"

    return INITIAL_PERSONA_PROMPT, REFINED_SYSTEM_PROMPT

async def get_active_persona(user: models.User, db: AsyncSession) -> str:
    """
    Resolve active persona prompt based on the priority:
    1. Custom Instruction (User custom override prompt)
    2. Dynamic System Prompt (Evolves from INITIAL to REFINED after 5 turns)
    """
    if user.custom_instruction:
        return user.custom_instruction
    
    from sqlalchemy import func
    turn_count_query = select(func.count(models.ChatHistory.id)).where(models.ChatHistory.user_id == user.id)
    turn_count_res = await db.execute(turn_count_query)
    turn_count = turn_count_res.scalar() or 0

    initial_prompt, refined_prompt = await get_persona_prompts(user, db)
    
    if turn_count > 5:
        return refined_prompt
    return initial_prompt

async def refine_persona_prompt(user: models.User, db: AsyncSession, recent_history_list: list) -> str:
    """
    Refines the user's conversational preferences and interest context based on recent history.
    Keeps the output extremely concise, natural, and adaptive, avoiding rigid rules.
    """
    # 1. Get original persona base prompt
    p_query = select(models.Persona).where(models.Persona.cluster_id == user.cluster_id)
    p_result = await db.execute(p_query)
    persona_obj = p_result.scalar_one_or_none()
    original_persona = persona_obj.base_prompt if persona_obj else "Bạn là trợ lý đọc báo thông minh."
    
    # 2. Get current refined preferences
    current_preferences = user.persona_prompt if user.persona_prompt else "Chưa ghi nhận sở thích hội thoại đặc biệt nào."
    
    # 3. Concatenate recent history from USER-side signals only
    history_text = "\n".join([
        f"- Yêu cầu/Câu hỏi của độc giả: {h['message']}" 
        for h in recent_history_list[-7:] # Focus on last 7 user messages to detect recurring patterns
        if h.get('message')
    ])
    
    # 4. Build refinement instruction prompt focusing purely on User-Side Signals
    system_instruction = f"""Bạn là chuyên gia thiết kế chỉ dẫn hành vi.
Nhiệm vụ của bạn là phân tích danh sách các câu hỏi, yêu cầu và hành vi gần đây của độc giả để cập nhật/tinh chỉnh một chỉ dẫn hành vi cực kỳ ngắn gọn (2-3 câu, dưới 80 từ) mô tả cách trợ lý nên tự nhiên thích ứng phong cách phản hồi cho độc giả này trong tương lai.

=== CHỈ DẪN HÀNH VI HIỆN TẠI ===
{current_preferences}

=== DANH SÁCH CÁC YÊU CẦU & CÂU HỎI GẦN ĐÂY CỦA ĐỘC GIẢ ===
{history_text}

=== TINH CHỈNH CHỈ DẪN HÀNH VI ===
1. Nhận diện sở thích thông qua các câu hỏi lặp đi lặp lại về một chủ đề (như tài chính, AI), các từ khóa thể hiện yêu cầu định dạng (như "tóm tắt", "liệt kê", "ngắn gọn") hoặc các phản hồi ý kiến trực tiếp.
2. TRÁNH PHẢN HỒI VÒNG LẶP: Tuyệt đối không giả định phong cách của câu trả lời trước đó của trợ lý là sở thích của độc giả trừ khi độc giả tự mình đưa ra yêu cầu rõ ràng.
3. VIẾT CHỈ DẪN HÀNH VI: Hướng dẫn trực tiếp cho trợ lý cách trả lời độc giả này khi thảo luận, tuyệt đối không viết mô tả tĩnh về hồ sơ của người dùng.
4. GIỮ CHO linh hoạt và mềm mại, tránh các quy tắc cứng nhắc hoặc các ràng buộc tuyệt đối.
5. Phản hồi CHỈ chứa đoạn văn chỉ dẫn hành vi mới bằng tiếng Việt, không thêm bất kỳ lời chào, lời dẫn hay giải thích nào khác.

Hãy viết đoạn chỉ dẫn hành vi mới:"""
    
    try:
        refined_prompt = await chat_vllm(
            messages=[{"role": "user", "content": "Tạo phân tích đúc kết thói quen độc giả mới."}],
            system=system_instruction
        )
        refined_prompt = refined_prompt.strip()
        
        # Save refined prompt back to user profile
        user.persona_prompt = refined_prompt
        await db.commit()
        await db.refresh(user)
        print(f"Refinement Successful for user '{user.username}': Evolved preferences updated.")
        return refined_prompt
    except Exception as e:
        print(f"Refinement Error for user '{user.username}': {e}")
        return current_preferences
