import json
import time
import re
import statistics

import psycopg2
from psycopg2.extras import RealDictCursor

from sentence_transformers import SentenceTransformer
import requests

# =========================================================
# CONFIG
# =========================================================

POSTGRES_CONFIG = {
    'host': '10.6.21.3',
    'database': 'optimize',
    'user': 'kietcorn',
    'password': 'kiietqo9204'
}

VLLM_URL = 'http://10.6.21.3:8888/v1/chat/completions'
MODEL = 'Qwen/Qwen2.5-7B-Instruct-AWQ'

TOP_K = 5
TEMPERATURE = 0.4
TOP_P = 0.8
MAX_TOKENS = 1024

# =========================================================
# LOAD MODEL
# =========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    'intfloat/multilingual-e5-base'
)

# Warmup
embedding_model.encode(["warmup"])

print("✓ Embedding model loaded\n")

# =========================================================
# TEST QUESTIONS
# =========================================================

test_questions = [
    "Phố Wall đã đưa ra dự đoán gì về giá vàng thế giới?",
    "Tại sao xác thực sinh trắc học có thể hạn chế tình trạng mượn tài khoản giao dịch?",
    "Thái Lan dùng phương pháp gì để siết chất lượng sầu riêng?",
    "Vincom Retail hợp tác với bên nào để mở tổ hợp giải trí Hàn Quốc?",
    "Quốc gia nào có thành phố hạnh phúc nhất Đông Nam Á?",
    "Tại sao nhiều sinh viên hiện nay mông lung việc làm khi ra trường?",
    "Các nhà đầu tư lo lắng gì về sự bất tiện của việc xác thực sinh trắc học?",
    "Những khó khăn trong việc triển khai xác thực sinh trắc học là gì?",
]

# =========================================================
# USER PROMPTS
# =========================================================

USER_PROMPTS = [
    {
        "name": "Journalist",
        "system": """
Bạn là nhà báo tài chính chuyên nghiệp.

Yêu cầu:
- ngắn gọn
- khách quan
- giàu thông tin
- không lan man
"""
    },

    {
        "name": "Professor",
        "system": """
Bạn là giảng viên đại học.

Yêu cầu:
- giải thích logic
- dễ hiểu
- có phân tích nguyên nhân-kết quả
- mang tính giáo dục
"""
    },

    {
        "name": "GenZ",
        "system": """
Bạn là content creator Gen Z.

Yêu cầu:
- tự nhiên
- dễ hiểu
- sinh động
- có thể dùng emoji nhẹ
- tránh quá học thuật
"""
    },

    {
        "name": "Financial Analyst",
        "system": """
Bạn là chuyên gia phân tích đầu tư.

Yêu cầu:
- tập trung tác động kinh tế
- đánh giá rủi ro/lợi ích
- chuyên nghiệp
- có góc nhìn thị trường
"""
    },

    {
        "name": "TV Host",
        "system": """
Bạn là MC bản tin truyền hình.

Yêu cầu:
- hấp dẫn
- dẫn dắt tốt
- mạch lạc
- dễ nghe khi đọc thành tiếng
"""
    },

    {
        "name": "Critic",
        "system": """
Bạn là chuyên gia phản biện.

Yêu cầu:
- phân tích đa chiều
- nêu cả ưu và nhược điểm
- chỉ ra hạn chế tiềm ẩn
"""
    },

    {
        "name": "Researcher",
        "system": """
Bạn là researcher học thuật.

Yêu cầu:
- trung tính
- chặt chẽ
- có cấu trúc
- ưu tiên độ chính xác
"""
    },

    {
        "name": "Minimal Assistant",
        "system": """
Bạn là AI assistant tối giản.

Yêu cầu:
- cực ngắn gọn
- tối đa 3 câu
- chỉ giữ ý chính
"""
    }
]

# =========================================================
# DATABASE
# =========================================================

def connect_postgres():
    return psycopg2.connect(
        host=POSTGRES_CONFIG['host'],
        database=POSTGRES_CONFIG['database'],
        user=POSTGRES_CONFIG['user'],
        password=POSTGRES_CONFIG['password']
    )

# =========================================================
# EMBEDDING
# =========================================================

def embed_question(question):

    embedding = embedding_model.encode(
        question,
        convert_to_tensor=False
    )

    return embedding.tolist()

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_similar_chunks(
    conn,
    question_embedding,
    top_k=5
):

    cursor = conn.cursor(
        cursor_factory=RealDictCursor
    )

    query = """
        SELECT
            id,
            text AS content,
            metadata,
            1 - (embedding <=> %s::vector) AS similarity
        FROM dangtuan
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    cursor.execute(
        query,
        (
            question_embedding,
            question_embedding,
            top_k
        )
    )

    results = cursor.fetchall()

    cursor.close()

    return results

# =========================================================
# METRICS
# =========================================================

def calculate_metrics(text):

    words = re.findall(
        r'\w+',
        text.lower()
    )

    unique_words = set(words)

    word_count = len(words)

    unique_ratio = (
        len(unique_words) / word_count
        if word_count > 0 else 0
    )

    sentence_count = max(
        1,
        len(re.findall(r'[.!?]+', text))
    )

    avg_sentence_length = (
        word_count / sentence_count
    )

    emoji_count = len(
        re.findall(
            r'[\U00010000-\U0010ffff]',
            text
        )
    )

    bullet_count = len(
        re.findall(
            r'^\s*[-•*]',
            text,
            re.MULTILINE
        )
    )

    markdown_count = len(
        re.findall(
            r'[#*_`]',
            text
        )
    )

    uppercase_ratio = (
        sum(
            1 for c in text
            if c.isupper()
        ) / max(1, len(text))
    )

    return {
        "word_count": word_count,
        "unique_word_ratio": round(unique_ratio, 3),
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "emoji_count": emoji_count,
        "bullet_count": bullet_count,
        "markdown_symbols": markdown_count,
        "uppercase_ratio": round(uppercase_ratio, 3),
    }

# =========================================================
# QWEN QUERY
# =========================================================

def query_qwen(
    system_prompt,
    context,
    question
):

    max_context_len = 5000

    if len(context) > max_context_len:
        context = (
            context[:max_context_len]
            + "...[truncated]"
        )

    user_prompt = f"""
Dựa trên context dưới đây, hãy trả lời câu hỏi.

Context:
{context}

Question:
{question}
"""

    start_time = time.time()

    response = requests.post(
        VLLM_URL,
        json={
            'model': MODEL,
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': user_prompt
                }
            ],
            'temperature': TEMPERATURE,
            'top_p': TOP_P,
            'max_tokens': MAX_TOKENS
        },
        timeout=120
    )

    latency = time.time() - start_time

    response.raise_for_status()

    answer = (
        response.json()['choices'][0]
        ['message']['content']
    )

    return answer, latency

# =========================================================
# MAIN TEST
# =========================================================

print("=" * 100)
print("RAG USER PROMPT STYLE TEST")
print("=" * 100)

try:

    conn = connect_postgres()

    print("\n✓ Connected to Postgres")

    all_metrics = []

    for idx, question in enumerate(test_questions):

        print("\n" + "=" * 100)
        print(f"QUESTION {idx + 1}")
        print("=" * 100)

        user_prompt_config = USER_PROMPTS[idx]

        print(
            f"\n🎭 USER PROMPT: "
            f"{user_prompt_config['name']}"
        )

        print(f"❓ QUESTION: {question}")

        # -------------------------------------------------
        # EMBED QUESTION
        # -------------------------------------------------

        embed_start = time.time()

        q_embedding = embed_question(question)

        embed_latency = (
            time.time() - embed_start
        )

        # -------------------------------------------------
        # RETRIEVE
        # -------------------------------------------------

        retrieval_start = time.time()

        chunks = retrieve_similar_chunks(
            conn,
            q_embedding,
            top_k=TOP_K
        )

        retrieval_latency = (
            time.time() - retrieval_start
        )

        similarities = [
            float(chunk['similarity'])
            for chunk in chunks
        ]

        avg_similarity = round(
            statistics.mean(similarities),
            4
        )

        # -------------------------------------------------
        # CONTEXT
        # -------------------------------------------------

        context = "\n\n---\n\n".join([

            f"[{json.loads(chunk['metadata']).get('title', 'N/A') if isinstance(chunk['metadata'], str) else chunk['metadata'].get('title', 'N/A')}]\n{chunk['content']}"

            for chunk in chunks
        ])

        # -------------------------------------------------
        # GENERATE
        # -------------------------------------------------

        answer, generation_latency = query_qwen(
            user_prompt_config['system'],
            context,
            question
        )

        metrics = calculate_metrics(answer)

        all_metrics.append({

            "user_prompt":
                user_prompt_config['name'],

            "question":
                question,

            "embed_latency":
                round(embed_latency, 3),

            "retrieval_latency":
                round(retrieval_latency, 3),

            "generation_latency":
                round(generation_latency, 3),

            "avg_similarity":
                avg_similarity,

            **metrics
        })

        # -------------------------------------------------
        # OUTPUT
        # -------------------------------------------------

        print("\n" + "-" * 100)
        print("📊 RETRIEVAL")
        print("-" * 100)

        print(f"Top-K: {TOP_K}")

        print(
            f"Average Similarity: "
            f"{avg_similarity}"
        )

        print("\nTop Retrieved Documents:")

        for i, chunk in enumerate(chunks):

            metadata = (
                json.loads(chunk['metadata'])
                if isinstance(
                    chunk['metadata'],
                    str
                )
                else chunk['metadata']
            )

            print(
                f"{i+1}. "
                f"{metadata.get('title', 'N/A')[:80]}"
                f" | sim="
                f"{round(float(chunk['similarity']), 4)}"
            )

        print("\n" + "-" * 100)
        print("💬 ANSWER")
        print("-" * 100)

        print(answer)

        print("\n" + "-" * 100)
        print("📈 METRICS")
        print("-" * 100)

        print(
            json.dumps(
                metrics,
                indent=2,
                ensure_ascii=False
            )
        )

        print("\n⏱ LATENCY")

        print(
            f"Embedding: "
            f"{round(embed_latency, 3)}s"
        )

        print(
            f"Retrieval: "
            f"{round(retrieval_latency, 3)}s"
        )

        print(
            f"Generation: "
            f"{round(generation_latency, 3)}s"
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    for metric in all_metrics:

        print(
            f"\n[{metric['user_prompt']}] "
            f"words={metric['word_count']} | "
            f"unique_ratio="
            f"{metric['unique_word_ratio']} | "
            f"emoji={metric['emoji_count']} | "
            f"bullets={metric['bullet_count']} | "
            f"gen_time="
            f"{metric['generation_latency']}s"
        )

    conn.close()

    print("\n✓ TEST COMPLETED")

except Exception as e:

    print(f"\n❌ ERROR: {str(e)}")