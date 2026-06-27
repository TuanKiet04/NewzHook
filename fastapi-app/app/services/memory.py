#app/services/memory.py
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text

OLLAMA_URL = "http://10.6.21.3:11435"
OLLAMA_MODEL = "phi3-mini"


async def get_chat_history(user_id: int, db: AsyncSession, limit: int = 5) -> list:
    """Lấy lịch sử chat gần nhất của user, trả về dạng messages list."""
    from app.models import ChatHistory

    past_q = (
        select(ChatHistory)
        .where(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(limit)
    )
    past_res = await db.execute(past_q)
    past = past_res.scalars().all()

    history = []
    for h in reversed(past):
        history.append({"role": "user", "content": h.message})
        history.append({"role": "assistant", "content": h.response})

    return history


async def get_user_summary(user_id: int, db: AsyncSession) -> str:
    """Lấy summary các phiên cũ của user từ DB."""
    result = await db.execute(
        text("SELECT summary FROM user_summaries WHERE user_id = :uid ORDER BY created_at DESC LIMIT 1"),
        {"uid": user_id}
    )
    row = result.fetchone()
    return row.summary if row else ""


async def save_chat(user_id: int, message: str, response: str, db: AsyncSession):
    """Lưu 1 lượt chat vào DB."""
    from app.models import ChatHistory

    db.add(ChatHistory(
        user_id=user_id,
        message=message,
        response=response
    ))
    await db.commit()


async def summarize_and_save(user_id: int, db: AsyncSession):
    """
    Tóm tắt toàn bộ lịch sử chat hiện tại của user bằng LLM,
    lưu vào bảng user_summaries, rồi xóa history cũ.
    Gọi hàm này sau mỗi N lượt chat (ví dụ 10 lượt).
    """
    from app.models import ChatHistory

    # Lấy toàn bộ history hiện tại
    past_q = (
        select(ChatHistory)
        .where(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.asc())
    )
    past_res = await db.execute(past_q)
    all_history = past_res.scalars().all()

    if not all_history:
        return

    # Ghép thành đoạn hội thoại để tóm tắt
    conversation_text = ""
    for h in all_history:
        conversation_text += f"User: {h.message}\nAssistant: {h.response}\n\n"

    summary_prompt = f"""Tóm tắt ngắn gọn cuộc hội thoại sau trong 3-5 câu. 
Tập trung vào: chủ đề người dùng quan tâm, tone/cách nói chuyện của họ, và các thông tin quan trọng đã được đề cập.

Hội thoại:
{conversation_text}

Tóm tắt:"""

    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": summary_prompt}],
                "max_tokens": 256,
                "temperature": 0.3
            }
        )

    summary = res.json()["choices"][0]["message"]["content"].strip()

    # Lưu summary vào DB
    await db.execute(
        text("""
            INSERT INTO user_summaries (user_id, summary, created_at)
            VALUES (:uid, :summary, NOW())
        """),
        {"uid": user_id, "summary": summary}
    )

    # Xóa history cũ sau khi đã tóm tắt
    await db.execute(
        text("DELETE FROM chat_history WHERE user_id = :uid"),
        {"uid": user_id}
    )

    await db.commit()


async def maybe_summarize(user_id: int, db: AsyncSession, threshold: int = 10):
    """
    Kiểm tra nếu history >= threshold thì tóm tắt và reset.
    Gọi hàm này sau mỗi lần save_chat.
    """
    from app.models import ChatHistory

    count_res = await db.execute(
        text("SELECT COUNT(*) FROM chat_history WHERE user_id = :uid"),
        {"uid": user_id}
    )
    count = count_res.scalar()

    if count >= threshold:
        await summarize_and_save(user_id, db)