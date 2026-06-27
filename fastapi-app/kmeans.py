#kmeans.py
import psycopg2
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from collections import Counter
import umap
import joblib, os

os.makedirs('models', exist_ok=True)

DATABASE_URL = 'postgresql://kietcorn:kiietqo9204@10.6.21.3:5432/optimize'

# ─── 1. Load embeddings ───────────────────────────────────────────────────────
print('Loading embeddings...')
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("SELECT metadata, embedding FROM embed_context ORDER BY id;")
rows = cur.fetchall()

embeddings, topic_labels, titles, published_ats, article_ids = [], [], [], [], []
for row in rows:
    raw = row[1]
    vec = json.loads(raw) if isinstance(raw, str) else list(raw)
    embeddings.append(vec)
    meta = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    topic_labels.append(meta.get('topic', 'Unknown'))
    titles.append(meta.get('title', 'No title'))
    published_ats.append(meta.get('published_at', ''))
    article_ids.append(meta.get('article_id', ''))

X = np.array(embeddings, dtype=np.float32)
print(f'Loaded {X.shape[0]} vectors, dim={X.shape[1]}')
X = normalize(X, norm='l2')
unique_topics = list(set(topic_labels))

print('\nTopic distribution:')
topic_counts = Counter(topic_labels)
for t, c in sorted(topic_counts.items(), key=lambda x: -x[1]):
    print(f'  {t}: {c}')

# ─── 2. PCA ───────────────────────────────────────────────────────────────────
print('\nRunning PCA...')
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X)
explained = np.sum(pca.explained_variance_ratio_) * 100
print(f'Explained variance with 50 components: {explained:.1f}%')

# Topic centroids trên PCA space
# Dùng 5 bài mới nhất của mỗi topic thay vì mean toàn bộ
N_RECENT = 20
topic_centroids = {}
for t in unique_topics:
    indices = [i for i, label in enumerate(topic_labels) if label == t]
    # Sort theo published_at mới nhất
    indices_sorted = sorted(indices, key=lambda i: published_ats[i], reverse=True)
    top_indices = indices_sorted[:N_RECENT]
    vecs = X_pca[top_indices]
    centroid = np.mean(vecs, axis=0)
    topic_centroids[t] = centroid / np.linalg.norm(centroid)
    print(f'  {t}: dùng {len(top_indices)} bài mới nhất')

# ─── 3. Sample weights để compensate imbalance ────────────────────────────────
print('\nComputing sample weights...')
total = len(topic_labels)
n_classes = len(unique_topics)
sample_weights = np.array([
    total / (n_classes * topic_counts[t])
    for t in topic_labels
])
print('Weights per topic:')
for t in unique_topics:
    w = total / (n_classes * topic_counts[t])
    print(f'  {t}: {w:.4f}')

# ─── 4. KMeans với cosine-based evaluation ───────────────────────────────────
print('\nSemantic Cluster Quality Scores:')

scores = {}
cluster_stats = {}

# Normalize PCA output để cosine metric hoạt động đúng
X_eval = normalize(X_pca, norm='l2')

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(
        X_pca,
        sample_weight=sample_weights
    )
    sil_score = silhouette_score(
        X_eval,
        labels,
        metric='cosine'
    )
    dominant_ratios = []

    for cluster_id in range(k):
        indices = np.where(labels == cluster_id)[0]

        if len(indices) == 0:
            continue

        cluster_topics = [topic_labels[i] for i in indices]
        counts = Counter(cluster_topics)

        dominant_count = counts.most_common(1)[0][1]
        dominant_ratio = dominant_count / len(indices)

        dominant_ratios.append(dominant_ratio)

    purity_score = np.mean(dominant_ratios)
    quality_score = (
        sil_score * 0.4 +
        purity_score * 0.6
    ) * 100

    scores[k] = quality_score

    cluster_stats[k] = {
        'silhouette': sil_score,
        'purity': purity_score,
        'quality': quality_score
    }

    print(
        f'  K={k}: '
        f'Quality={quality_score:.1f}/100 | '
        f'Sil={sil_score:.4f} | '
        f'Purity={purity_score:.4f}'
    )

# ─────────────────────────────────────────────────────────────
# Best K
# ─────────────────────────────────────────────────────────────
best_k = max(scores, key=scores.get)

best_stat = cluster_stats[best_k]

print('\n────────────────────────────────────')
print(f'Best K = {best_k}')
print(f'Semantic Quality Score : {best_stat["quality"]:.1f}/100')
print(f'Silhouette (cosine)   : {best_stat["silhouette"]:.4f}')
print(f'Topic Purity          : {best_stat["purity"]:.4f}')
print('────────────────────────────────────')
final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

final_model.fit(
    X_pca,
    sample_weight=sample_weights
)

final_labels = final_model.predict(X_pca)

print('\nPersisting final labels into raw_data table in PostgreSQL...')
try:
    # 1. Add cluster_id column to raw_data if not exists
    cur.execute("ALTER TABLE public.raw_data ADD COLUMN IF NOT EXISTS cluster_id INTEGER;")
    print("Column 'cluster_id' verified/added in 'raw_data' table.")
    
    # 2. Update cluster_id for each article
    update_data = []
    for art_id, label in zip(article_ids, final_labels):
        if art_id:
            update_data.append((int(label), art_id))
            
    from psycopg2.extras import execute_batch
    execute_batch(cur, "UPDATE public.raw_data SET cluster_id = %s WHERE id = %s;", update_data)
    conn.commit()
    print(f"Persisted {len(update_data)} article cluster assignments into raw_data.")
except Exception as e:
    conn.rollback()
    print(f"Error persisting article cluster labels: {e}")

centroids_pca = final_model.cluster_centers_

# Normalize centroids cho cosine similarity
centroids_norm = centroids_pca / np.linalg.norm(
    centroids_pca,
    axis=1,
    keepdims=True
)

# Save
joblib.dump(final_model, 'models/kmeans_model.pkl')
joblib.dump(pca, 'models/pca_model.pkl')
joblib.dump(topic_centroids, 'models/topic_centroids.pkl')
joblib.dump(centroids_pca, 'models/centroids_pca.pkl')
joblib.dump(centroids_norm, 'models/centroids_norm.pkl')

print('\nModels saved to models/')

# ─── 5. UMAP để visualize ─────────────────────────────────────────────────────
print('\nRunning UMAP for visualization only...')
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
X_2d = reducer.fit_transform(X_pca)

trust = trustworthiness(X_pca, X_2d, n_neighbors=15)
print(f'Trustworthiness score: {trust:.4f}')

# ─── 6. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for cluster_id in range(best_k):
    mask = final_labels == cluster_id
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[cluster_id]],
                    label=f'Cluster {cluster_id}', alpha=0.6, s=20)
axes[0].set_title(f'K-Means Clusters (K={best_k}) - UMAP 2D\nTrustworthiness: {trust:.4f}')
axes[0].legend(markerscale=2)
axes[0].grid(True, alpha=0.3)

topic_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_topics)))
topic_color_map = {t: topic_colors[i] for i, t in enumerate(unique_topics)}
for topic in unique_topics:
    mask = np.array([t == topic for t in topic_labels])
    axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[topic_color_map[topic]],
                    label=topic, alpha=0.6, s=20)
axes[1].set_title('Colored by Topic (original distribution)')
axes[1].legend(markerscale=2, fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_umap_final.png', dpi=150, bbox_inches='tight')
print('Saved: kmeans_umap_final.png')

# ─── 7. Cluster details ───────────────────────────────────────────────────────
print(f'\n--- Cluster details (K={best_k}) ---')
for cluster_id in range(best_k):
    indices = np.where(final_labels == cluster_id)[0]
    cluster_topic_counts = Counter([topic_labels[i] for i in indices])
    dominant = cluster_topic_counts.most_common(1)[0]
    print(f'\nCluster {cluster_id} ({len(indices)} articles): {dict(cluster_topic_counts)}')
    print(f'  → Dominant: {dominant[0]} ({dominant[1]/len(indices)*100:.1f}%)')

# ─── 8. Test assign user ──────────────────────────────────────────────────────
topic_map = {
    1: 'Thời Sự', 2: 'Pháp Luật', 3: 'Công Nghệ',
    4: 'Kinh Tế', 5: 'Thể Thao', 6: 'Giáo Dục'
}

print('\n--- Test assign user (nhập q để thoát) ---')
for k, v in topic_map.items():
    print(f'  {k}. {v}')

while True:
    selected = input('\nChọn 3-4 topic (ví dụ: 1 3 5) hoặc q để thoát: ').strip()
    if selected.lower() == 'q':
        break

    try:
        selected_topics = [topic_map[int(s)] for s in selected.split()]
        if not (3 <= len(selected_topics) <= 4):
            print('Vui lòng chọn 3 hoặc 4 topic.')
            continue
    except (KeyError, ValueError):
        print('Input không hợp lệ.')
        continue

    print(f'Bạn chọn: {selected_topics}')

    # Cosine similarity để assign
    user_vec = np.mean([topic_centroids[t] for t in selected_topics], axis=0)
    user_vec /= np.linalg.norm(user_vec)

    similarities = centroids_norm @ user_vec
    assigned_cluster = np.argmax(similarities)
    cluster_indices = np.where(final_labels == assigned_cluster)[0]
    cluster_topic_counts = Counter([topic_labels[i] for i in cluster_indices])
    dominant = cluster_topic_counts.most_common(1)[0]

    print(f'\nUser được assign vào: Cluster {assigned_cluster} (similarity={similarities[assigned_cluster]:.4f})')
    print(f'Topics trong cluster: {dict(cluster_topic_counts)}')
    print(f'Dominant topic: {dominant[0]} ({dominant[1]/len(cluster_indices)*100:.1f}%)')

    print(f'\nMột số bài báo trong Cluster {assigned_cluster}:')
    sample_indices = np.random.choice(cluster_indices, size=min(3, len(cluster_indices)), replace=False)
    for idx in sample_indices:
        print(f'  [{topic_labels[idx]}] {titles[idx][:80]}')

    print(f'Similarity scores tất cả clusters: {[f"C{i}:{s:.4f}" for i, s in enumerate(similarities)]}')