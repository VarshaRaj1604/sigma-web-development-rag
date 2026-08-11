import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sigma Web Development AI Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_FILE = "embeddings.joblib"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

TOP_RESULTS = 5


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    with st.spinner("Loading Qwen embedding model..."):

        model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

    return model


model = load_embedding_model()


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_data
def load_embeddings():

    df = joblib.load(EMBEDDINGS_FILE)

    return df


df = load_embeddings()


# ============================================================
# FORMAT TIMESTAMP
# ============================================================

def format_time(seconds):

    try:
        seconds = float(seconds)

    except (ValueError, TypeError):
        return "00:00"

    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    seconds = int(
        seconds % 60
    )

    if hours > 0:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


# ============================================================
# SEARCH COURSE
# ============================================================

def search_course(query, top_k=5):

    # --------------------------------------------------------
    # CREATE QUERY EMBEDDING
    # --------------------------------------------------------

    question_embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # GET STORED EMBEDDINGS
    # --------------------------------------------------------

    embedding_matrix = np.vstack(
        df["embedding"].values
    ).astype(np.float32)


    # --------------------------------------------------------
    # NORMALIZE STORED EMBEDDINGS
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embedding_matrix,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    embedding_matrix = (
        embedding_matrix / norms
    )


    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    similarities = np.dot(
        embedding_matrix,
        question_embedding
    )


    # --------------------------------------------------------
    # GET TOP RESULTS
    # --------------------------------------------------------

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]


    results = df.iloc[
        top_indices
    ].copy()


    results["similarity"] = (
        similarities[top_indices]
    )


    return results


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🎓 Sigma Web Development AI Assistant"
)

st.write(
    "Ask questions about the Sigma Web Development "
    "Course and find the relevant video, timestamp "
    "and course content."
)


# ============================================================
# COURSE INFORMATION
# ============================================================

st.success(
    f"Loaded {len(df)} course chunks."
)


# ============================================================
# QUESTION INPUT
# ============================================================

query = st.text_input(
    "🔎 Ask a question",
    placeholder="Example: What is HTML?"
)


# ============================================================
# SEARCH BUTTON
# ============================================================

search_button = st.button(
    "🔍 Search Course"
)


# ============================================================
# SEARCH
# ============================================================

if search_button:

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the course..."
        ):

            try:

                results = search_course(
                    query,
                    TOP_RESULTS
                )


                # =================================================
                # SEARCH COMPLETED
                # =================================================

                st.success(
                    "Search completed!"
                )


                st.subheader(
                    "📚 Most Relevant Course Content"
                )


                # =================================================
                # DISPLAY RESULTS
                # =================================================

                for rank, (_, row) in enumerate(
                    results.iterrows(),
                    start=1
                ):

                    st.markdown(
                        f"## 📚 Result {rank}"
                    )


                    # ---------------------------------------------
                    # SIMILARITY
                    # ---------------------------------------------

                    st.write(
                        f"**Similarity:** "
                        f"{row['similarity']:.4f}"
                    )


                    # ---------------------------------------------
                    # VIDEO TITLE
                    # ---------------------------------------------

                    title = row.get(
                        "title",
                        "Not available"
                    )

                    st.write(
                        f"🎬 **Video:** {title}"
                    )


                    # ---------------------------------------------
                    # VIDEO NUMBER
                    # ---------------------------------------------

                    number = row.get(
                        "number",
                        "Not available"
                    )

                    st.write(
                        f"🔢 **Video Number:** {number}"
                    )


                    # ---------------------------------------------
                    # TIMESTAMP
                    # ---------------------------------------------

                    start_time = row.get(
                        "start",
                        row.get("start_time", 0)
                    )

                    end_time = row.get(
                        "end",
                        row.get("end_time", 0)
                    )


                    start_formatted = format_time(
                        start_time
                    )

                    end_formatted = format_time(
                        end_time
                    )


                    st.success(
                        f"⏱️ **Timestamp: "
                        f"{start_formatted} → "
                        f"{end_formatted}**"
                    )


                    # ---------------------------------------------
                    # COURSE CONTENT
                    # ---------------------------------------------

                    st.write(
                        "🎥 **Course Content:**"
                    )

                    text = row.get(
                        "text",
                        "No text available"
                    )

                    st.write(text)


                    # ---------------------------------------------
                    # VIDEO POSITION
                    # ---------------------------------------------

                    st.info(
                        f"▶️ Go to approximately "
                        f"**{start_formatted}** "
                        f"in this video."
                    )


                    # ---------------------------------------------
                    # SEPARATOR
                    # ---------------------------------------------

                    st.divider()


            except Exception as e:

                st.error(
                    "Something went wrong while searching."
                )

                st.exception(e)