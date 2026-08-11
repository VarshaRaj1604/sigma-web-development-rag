import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_FILE = "embeddings.joblib"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

TOP_K = 5


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("Loading course embeddings...")

df = joblib.load(EMBEDDINGS_FILE)

print("Embeddings loaded!")
print("Total chunks:", len(df))


# ============================================================
# CREATE EMBEDDING FOR QUESTION
# ============================================================

def create_question_embedding(question):

    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding


# ============================================================
# SEARCH COURSE
# ============================================================

def search_course(question, top_k=TOP_K):

    # Create question embedding
    question_embedding = create_question_embedding(
        question
    )

    # Convert stored embeddings to matrix
    embedding_matrix = np.vstack(
        df["embedding"].values
    )

    # Check dimensions
    print(
        "Course embedding dimension:",
        embedding_matrix.shape[1]
    )

    print(
        "Question embedding dimension:",
        question_embedding.shape[1]
    )

    # Calculate similarity
    similarities = cosine_similarity(
        embedding_matrix,
        question_embedding
    ).flatten()

    # Get top results
    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    results = []

    for index in top_indices:

        row = df.iloc[index]

        results.append({
            "chunk_id": row["chunk_id"],
            "title": row.get("title", ""),
            "number": row.get("number", ""),
            "start": row.get("start", 0),
            "end": row.get("end", 0),
            "text": row["text"],
            "similarity": float(
                similarities[index]
            )
        })

    return results


# ============================================================
# FORMAT TIME
# ============================================================

def format_timestamp(seconds):

    seconds = float(seconds)

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes:02d}:{remaining_seconds:02d}"


# ============================================================
# ASK QUESTIONS
# ============================================================

while True:

    print()
    print("=" * 70)

    question = input(
        "Ask a question (type 'exit' to quit): "
    )

    if question.lower().strip() == "exit":

        print("Goodbye!")
        break

    if not question.strip():

        print("Please enter a question.")

        continue

    print()
    print("Searching course...")

    results = search_course(
        question,
        TOP_K
    )

    print()
    print("=" * 70)
    print("MOST RELEVANT COURSE CONTENT")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        start_time = format_timestamp(
            result["start"]
        )

        end_time = format_timestamp(
            result["end"]
        )

        print()
        print("-" * 70)

        print(
            f"Result {rank}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Video: "
            f"{result['title']}"
        )

        print(
            f"Video Number: "
            f"{result['number']}"
        )

        print(
            f"Timestamp: "
            f"{start_time} - {end_time}"
        )

        print()

        print(
            "Text:"
        )

        print(
            result["text"]
        )

    print()