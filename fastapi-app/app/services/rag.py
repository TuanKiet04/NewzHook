#app/services/rag.py
import json
import time
import asyncio
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

VLLM_URL = "http://10.6.21.3:8888"

# Lazily load embedding model to keep FastAPI startup instantaneous
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model 'intfloat/multilingual-e5-base'...")
        _embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')
        print("✓ Embedding model loaded")
    return _embedding_model

async def get_multilingual_e5_embedding(query: str) -> list:
    """
    Computes query embeddings using intfloat/multilingual-e5-base.
    Runs in an executor to avoid blocking the async event loop.
    """
    loop = asyncio.get_running_loop()
    model = get_embedding_model()
    
    def _encode():
        return model.encode(query, convert_to_tensor=False).tolist()
        
    return await loop.run_in_executor(None, _encode)

def _to_pgvector(embedding: list) -> str:
    return "[" + ",".join(map(str, embedding)) + "]"

async def get_rag_context(query: str, db: AsyncSession, distance_threshold: float = 0.95) -> str:
    """
    Standard context retrieval endpoint (keeps compatible signature).
    """
    try:
        embedding = await get_multilingual_e5_embedding(query)
        embedding_str = _to_pgvector(embedding)

        result = await db.execute(text("""
            SELECT text as content
            FROM dangtuan
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 5
        """), {"embedding": embedding_str})

        rows = result.all()
        chunks = [row.content for row in rows]
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"RAG context retrieval error: {e}")
        return ""

async def get_rag_context_with_observability(query: str, db: AsyncSession, top_k: int = 5) -> tuple[str, list[dict], float, float]:
    """
    Extended context retrieval with latency tracking, similarity scores, and metadata extraction.
    """
    # 1. Measure embedding latency
    embed_start = time.time()
    embedding = await get_multilingual_e5_embedding(query)
    embed_latency = time.time() - embed_start

    # 2. Measure database retrieval latency
    retrieval_start = time.time()
    embedding_str = _to_pgvector(embedding)

    result = await db.execute(text("""
        SELECT id, text as content, metadata,
               1 - (embedding <=> CAST(:embedding AS vector)) as similarity
        FROM dangtuan
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """), {"embedding": embedding_str, "top_k": top_k})

    rows = result.all()
    retrieval_latency = time.time() - retrieval_start

    # 3. Format chunks and extract source metadata
    formatted_chunks = []
    top_chunks = []
    for row in rows:
        meta = row.metadata
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        elif not isinstance(meta, dict):
            meta = {}

        title = meta.get("title", "N/A")
        url = meta.get("url", "#")
        similarity = float(row.similarity)

        formatted_chunks.append(f"[{title}]\n{row.content}")
        top_chunks.append({
            "title": title,
            "url": url,
            "similarity": similarity
        })

    context_text = "\n\n---\n\n".join(formatted_chunks) if formatted_chunks else ""
    return context_text, top_chunks, embed_latency, retrieval_latency

async def get_recommendations(query: str, db: AsyncSession, limit: int = 3) -> list:
    """
    Retrieve similar recommended articles from dangtuan table.
    Applies deduplication by article URL and skips the top 2-3 most similar articles.
    """
    try:
        embedding = await get_multilingual_e5_embedding(query)
        embedding_str = _to_pgvector(embedding)

        # Retrieve a larger pool (top 15) to guarantee enough unique articles after filtering
        result = await db.execute(text("""
            SELECT text, metadata,
                   1 - (embedding <=> CAST(:embedding AS vector)) as similarity
            FROM dangtuan
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 15
        """), {"embedding": embedding_str})

        unique_articles = []
        seen_urls = set()

        for row in result.all():
            meta = row.metadata
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            elif not isinstance(meta, dict):
                meta = {}
                
            title = meta.get("title", "N/A")
            url = meta.get("url", "#")
            
            # Normalize URL to avoid minor trailing slash discrepancies
            normalized_url = url.strip().rstrip('/')
            
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_articles.append({
                    "title": title,
                    "url": url
                })

        # Skip the top 2 most similar articles (which are likely already discussed in RAG context)
        # and select the next 'limit' distinct articles.
        final_recs = unique_articles[2 : 2 + limit]
        
        # Fallback in case unique article pool is too small after skipping
        if len(final_recs) < limit and len(unique_articles) > 0:
            final_recs = unique_articles[:limit]
            
        return final_recs
    except Exception as e:
        print(f"Error in recommendations: {e}")
        return []

async def get_history_recommendations(history_text: str, db: AsyncSession, limit: int = 3) -> list:
    """
    Retrieve recommendations based on conversation history using multilingual-e5.
    Applies deduplication by article URL and skips the top 2-3 most similar articles.
    """
    if not history_text.strip():
        return []
    try:
        embedding = await get_multilingual_e5_embedding(history_text)
        embedding_str = _to_pgvector(embedding)

        # Retrieve a larger pool (top 15) to guarantee enough unique articles after filtering
        result = await db.execute(text("""
            SELECT text, metadata,
                   1 - (embedding <=> CAST(:embedding AS vector)) as similarity
            FROM dangtuan
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 15
        """), {"embedding": embedding_str})

        unique_articles = []
        seen_urls = set()

        for row in result.all():
            meta = row.metadata
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            elif not isinstance(meta, dict):
                meta = {}
                
            title = meta.get("title", "N/A")
            url = meta.get("url", "#")
            
            # Normalize URL to avoid minor trailing slash discrepancies
            normalized_url = url.strip().rstrip('/')
            
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_articles.append({
                    "title": title,
                    "url": url
                })

        # Skip the top 2 most similar articles (which are likely already discussed in RAG context)
        # and select the next 'limit' distinct articles.
        final_recs = unique_articles[2 : 2 + limit]
        
        # Fallback in case unique article pool is too small after skipping
        if len(final_recs) < limit and len(unique_articles) > 0:
            final_recs = unique_articles[:limit]
            
        return final_recs
    except Exception as e:
        print(f"Error in history based recommendations: {e}")
        return []

def build_system_prompt(context: str, user_persona: str = None) -> str:
    if context:
        context_section = f"Dưới đây là các bài báo liên quan:\n\n{context}\n\nHãy trả lời DỰA TRÊN các bài báo trên."
    else:
        context_section = "Không có bài báo liên quan được tìm thấy trong hệ thống."

    persona_section = user_persona if user_persona else "Bạn là trợ lý tin tức thông minh."

    prompt = f"""{persona_section}

{context_section}
"""
    prompt += "\nNếu context không đủ thông tin, hãy kết hợp kiến thức của bạn để trả lời. Nếu hoàn toàn không biết, hãy thành thật nói không biết."
    return prompt