import streamlit as st
import requests
import json
import psycopg2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "10.6.21.3"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",   "optimize"),
    "user":     os.getenv("PG_USER", "kietcorn"),
    "password": os.getenv("PG_PASS", "kiietqo9204"),
}

OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://10.6.21.3:11435")
CHAT_URL = f"{OLLAMA_BASE}/api/chat"
EMBED_URL = f"{OLLAMA_BASE}/api/embeddings"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL   = "qwen2:7b"

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý thông minh, trả lời câu hỏi dựa trên thông tin được cung cấp. Hãy trả lời ngắn gọn, rõ ràng và chính xác."
)

BEHAVIOR_PROMPT = """VAI TRÒ:
Bạn là một trợ lý tin tức chuyên biệt, được tinh chỉnh riêng theo hành vi và hồ sơ rủi ro của người dùng. 

HỒ SƠ NGƯỜI DÙNG:
Dữ liệu lịch sử cho thấy người dùng này đặc biệt quan tâm tới:
- Công nghệ lõi: Trí tuệ nhân tạo (AI) và xu hướng công nghệ mới.
- Rủi ro tài chính: Các vụ gian lận, lừa đảo, cho vay nặng lãi, biến động chứng khoán.
- Pháp lý: Luật kinh tế và luật hình sự liên quan đến tội phạm công nghệ/tài chính.

Khi trả lời dựa trên [NGỮ CẢNH], bạn BẮT BUỘC phải bám sát các tiêu chí sau:
1. Luôn xoáy sâu vào các rủi ro thực tiễn, cảnh báo thiệt hại hoặc cơ hội phòng tránh.
2. Sử dụng thuật ngữ phân tích kỹ thuật khi nói về AI/Công nghệ, và giải thích các khía cạnh pháp luật một cách rành mạch, sắc bén.
3. Đi thẳng vào vấn đề cốt lõi liên quan đến tài sản, dữ liệu hoặc quyền lợi của người dùng trước, sau đó mới nêu chi tiết bối cảnh.
"""


# ---------- Load personas ----------

@st.cache_data
def load_personas():
    with open("personas.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["cluster_id"]: p for p in data}


# ---------- DB ----------

from psycopg2.pool import SimpleConnectionPool

# Khởi tạo pool (đặt ngoài các hàm cache)
@st.cache_resource
def get_db_pool():
    return SimpleConnectionPool(1, 10, **DB_CONFIG)

def retrieve(query: str, top_k: int = 5):
    vec = embed_query(query)
    # Đổi thành get_conn()
    with get_conn() as conn:
    try:
        with conn.cursor() as cur:
            # Chuyển list vec thành string để pgvector nhận diện chính xác
            vec_str = str(vec)
            cur.execute("""
                SELECT text, 
                       1 - (embedding <=> %s::vector) AS score
                FROM n8n_vectors
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (vec_str, vec_str, top_k))
            rows = cur.fetchall()
    finally:
        pool.putconn(conn) # Trả lại kết nối cho pool

    return [{"text": r[0], "score": round(r[1], 4)} for r in rows]


# Cáº¬P NHáº¬T HÃ€M RETRIEVE TRONG PYTHON
def retrieve(query: str, top_k: int = 5):
    vec = embed_query(query)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT text,
                       1 - (embedding <=> %s::vector) AS score
                FROM n8n_vectors
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (vec, vec, top_k))

            rows = cur.fetchall()

    return [
        {"text": r[0], "score": round(r[1], 4)}
        for r in rows
    ]


# ---------- Ollama ----------

@st.cache_data(show_spinner=False)
def embed_query(text):
    res = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    res.raise_for_status()
    return res.json()["embedding"]


def call_llm(system_prompt, context_chunks, user_query):
    # ÄÃƒ Sá»¬A: XÃ³a bá» [:400] Ä‘á»ƒ LLM Ä‘á»c toÃ n bá»™ ná»™i dung
    context  = "\n\n".join([c["text"] for c in context_chunks])
    user_msg = f"Thông tin tham khảo:\n{context}\n\nCâu hỏi: {user_query}"
    
    res = requests.post(CHAT_URL, json={
        "model":    LLM_MODEL,
        "stream":   False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
    }, timeout=120)
    res.raise_for_status()
    return res.json()["message"]["content"]


def run_rag(label, system_prompt, chunks, query):
    answer = call_llm(system_prompt, chunks, query)
    return label, answer


# ---------- UI ----------

st.set_page_config(page_title="RAG Persona Demo", layout="wide")
st.title("RAG Persona Demo")

personas = load_personas()

with st.sidebar:
    st.markdown("### Cài đặt")
    selected_id = st.radio(
        "Chọn Persona",
        options=list(personas.keys()),
        format_func=lambda x: personas[x]["persona"]["name"],
    )
    st.markdown("---")
    st.markdown(f"**Mô tả**")
    st.caption(personas[selected_id]["persona"]["description"])
    st.markdown("---")
    st.caption(f"Model: `{LLM_MODEL}`")
    st.caption(f"Embed: `{EMBED_MODEL}`")

persona_system_prompt = personas[selected_id]["persona"]["system_prompt"]
behavior_prompt       = BEHAVIOR_PROMPT

tab1, tab2, tab3, tab4 = st.tabs([
    "Baseline RAG",
    "RAG + Persona",
    "RAG + Behavior",
    "So sánh",
])


# ---------- Tab 1 ----------

with tab1:
    # ... (giá»¯ nguyÃªn pháº§n trÃªn)
    q1 = st.text_input("Câu hỏi", key="q1")
    if st.button("Gửi", key="btn1") and q1:
        with st.spinner("Đang xử lý..."):
            # ÄÃƒ Sá»¬A: Bá» dÃ²ng vec = embed_query(q1)
            chunks = retrieve(q1) 
            answer = call_llm(DEFAULT_SYSTEM_PROMPT, chunks, q1)
        st.markdown("**Trả lời:**")
        st.write(answer)
        with st.expander("Chunks retrieve"):
            for i, c in enumerate(chunks):
                st.markdown(f"**#{i+1}** (score: {c['score']})")
                st.write(c["text"][:300])
                st.divider()


# ---------- Tab 2 ----------

with tab2:
    st.subheader("RAG + Persona")
    st.caption(f"Persona: {personas[selected_id]['persona']['name']}")
    with st.expander("System Prompt", expanded=True):
        st.code(persona_system_prompt, language=None)

    q2 = st.text_input("Câu hỏi", key="q2")
    if st.button("Gửi", key="btn2") and q2:
        with st.spinner("Đang xử lý..."):
            chunks = retrieve(q2)   # ✅ truyền string
            answer = call_llm(persona_system_prompt, chunks, q2)

        st.markdown("**Trả lời:**")
        st.write(answer)
        with st.expander("Chunks retrieve"):
            for i, c in enumerate(chunks):
                st.markdown(f"**#{i+1}** (score: {c['score']})")
                st.write(c["text"][:300])
                st.divider()


# ---------- Tab 3 ----------

with tab3:
    st.subheader("RAG + Behavior Prompt")
    st.caption("Prompt sinh từ Behavior.")
    with st.expander("Behavior Prompt", expanded=True):
        st.code(behavior_prompt, language=None)

    q3 = st.text_input("Câu hỏi", key="q3")
    if st.button("Gửi", key="btn3") and q3:
        with st.spinner("Đang xử lý..."):
            chunks = retrieve(q3)   # ✅ truyền string
            answer = call_llm(behavior_prompt, chunks, q3)
        st.markdown("**Trả lời:**")
        st.write(answer)
        with st.expander("Chunks retrieve"):
            for i, c in enumerate(chunks):
                st.markdown(f"**#{i+1}** (score: {c['score']})")
                st.write(c["text"][:300])
                st.divider()


# ---------- Tab 4 ----------

with tab4:
    st.subheader("So sánh")
    st.caption("So sánh 3 phương pháp.")

    configs = [
        ("Baseline",  DEFAULT_SYSTEM_PROMPT),
        ("Persona",   persona_system_prompt),
        ("Behavior",  behavior_prompt),
    ]

    with st.expander("Xem System Prompts"):
        for label, sp in configs:
            st.markdown(f"**{label}**")
            st.code(sp, language=None)

    q4 = st.text_input("Câu hỏi", key="q4")
    if st.button("So sánh", key="btn4") and q4:
        with st.spinner("Đang retrieve..."):
            chunks = retrieve(q4) # Sá»­a tÆ°Æ¡ng tá»± á»Ÿ Ä‘Ã¢y

        results = {}
        with st.spinner("Äang cháº¡y song song 3 cáº¥u hÃ¬nh..."):
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(run_rag, label, sp, chunks, q4): label
                    for label, sp in configs
                }
                for future in as_completed(futures):
                    label, answer = future.result()
                    results[label] = answer

        col1, col2, col3 = st.columns(3)
        for col, (label, _) in zip([col1, col2, col3], configs):
            with col:
                st.markdown(f"**{label}**")
                st.write(results.get(label, ""))

        with st.expander("Chunks retrieve (dÃ¹ng chung)"):
            for i, c in enumerate(chunks):
                st.markdown(f"**#{i+1}** (score: {c['score']})")
                st.write(c["text"][:300])
                st.divider()
