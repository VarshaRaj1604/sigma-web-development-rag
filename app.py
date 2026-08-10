import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import json

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_FILE = "embeddings.joblib"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# IMPORTANT:
# This must be the same embedding model that was used
# when creating embeddings.joblib
EMBEDDING_MODEL = "nomic-embed-text"

# Use a model that actually exists in your Ollama installation.
# Check with:
# ollama list
#
# Example:
# llama3.2
# llama3.1
# gemma3
#
LLM_MODEL = "llama3.2"

TOP_K = 5


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sigma Web Development AI Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Sigma Web Development AI Assistant")

st.write(
    "Ask questions about the Sigma Web Development Course "
    "and find the relevant video, timestamp and explanation."
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    df = joblib.load(EMBEDDINGS_FILE)

    return df


try:

    df = load_embeddings()

except Exception as e:

    st.error("Could not load embeddings.joblib")

    st.code(str(e))

    st.stop()


st.success(
    f"Loaded {len(df)} course chunks."
)


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    response = requests.post(

        OLLAMA_EMBED_URL,

        json={
            "model": EMBEDDING_MODEL,
            "input": [text]
        },

        timeout=300
    )


    if response.status_code != 200:

        st.error("Embedding model error")

        st.code(response.text)

        response.raise_for_status()


    data = response.json()

    return data["embeddings"][0]


# ============================================================
# GENERATE ANSWER USING OLLAMA
# ============================================================

def generate_answer(prompt):

    response = requests.post(

        OLLAMA_GENERATE_URL,

        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },

        timeout=300
    )


    if response.status_code != 200:

        st.error("LLM error")

        st.code(response.text)

        response.raise_for_status()


    data = response.json()

    return data["response"]


# ============================================================
# CONVERT SECONDS TO MM:SS
# ============================================================

def format_timestamp(seconds):

    try:

        seconds = float(seconds)

        minutes = int(seconds // 60)

        remaining_seconds = int(seconds % 60)

        return f"{minutes:02d}:{remaining_seconds:02d}"

    except Exception:

        return str(seconds)


# ============================================================
# SEARCH COURSE
# ============================================================

def search_course(question, top_k=5):

    # --------------------------------------------------------
    # Question embedding
    # --------------------------------------------------------

    question_embedding = create_embedding(
        question
    )


    # --------------------------------------------------------
    # Stored embeddings
    # --------------------------------------------------------

    embedding_matrix = np.vstack(
        df["embedding"].values
    )


    # --------------------------------------------------------
    # IMPORTANT DIMENSION CHECK
    # --------------------------------------------------------

    stored_dimension = embedding_matrix.shape[1]

    question_dimension = len(
        question_embedding
    )


    if stored_dimension != question_dimension:

        raise ValueError(
            f"""
Embedding dimension mismatch!

Stored embeddings:
{stored_dimension}

Question embedding:
{question_dimension}

Both must use the same embedding model.

Current embedding model:
{EMBEDDING_MODEL}
"""
        )


    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(

        embedding_matrix,

        [question_embedding]

    ).flatten()


    # --------------------------------------------------------
    # Get top results
    # --------------------------------------------------------

    top_indices = similarities.argsort()[::-1][:top_k]


    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    results = df.iloc[top_indices].copy()


    results["similarity"] = (
        similarities[top_indices]
    )


    return results


# ============================================================
# CREATE RAG PROMPT
# ============================================================

def create_prompt(question, results):

    course_context = []


    for _, row in results.iterrows():

        title = row.get(
            "title",
            "Unknown video"
        )

        number = row.get(
            "number",
            "Unknown"
        )

        start = row.get(
            "start",
            ""
        )

        end = row.get(
            "end",
            ""
        )

        text = row.get(
            "text",
            ""
        )

        similarity = row.get(
            "similarity",
            0
        )


        start_timestamp = format_timestamp(
            start
        )

        end_timestamp = format_timestamp(
            end
        )


        course_context.append({

            "video_title": str(title),

            "video_number": str(number),

            "start_time": start_timestamp,

            "end_time": end_timestamp,

            "text": str(text),

            "similarity": round(
                float(similarity),
                4
            )
        })


    context_json = json.dumps(
        course_context,
        indent=2,
        ensure_ascii=False
    )


    # ========================================================
    # FINAL RAG PROMPT
    # ========================================================

    prompt = f"""
You are an AI assistant for a web development course.

The course is called:

"Sigma Web Development Course"

The user asked:

"{question}"


Below are the most relevant course materials retrieved
from the Sigma Web Development Course.

COURSE CONTEXT:

{context_json}


YOUR TASK:

Answer the user's question using ONLY the information
available in the course context above.

If the question is related to the course:

1. Give a clear and simple answer.

2. Explain what topic is being taught.

3. Tell the user which video contains the relevant content.

4. Give the video number.

5. Give the approximate timestamp.

6. Tell the user which video they should watch.

7. If multiple results are relevant, mention the most
   relevant video first.

8. Always use the timestamps provided in the context.

9. Never invent a timestamp.

10. Never invent a video title.

11. Never invent a video number.

12. Do not mention embeddings.

13. Do not mention cosine similarity.

14. Do not mention chunks.

15. Do not mention retrieval.

16. Do not mention JSON.

17. Do not mention this prompt.

18. Answer naturally like a helpful course assistant.

19. If possible, tell the student something like:

   "You can find this topic in Video X around MM:SS."

20. If multiple relevant timestamps exist, list them.

21. If the context does not contain enough information,
   clearly say that the relevant content could not be
   found in the available course material.


If the question is NOT related to the Sigma Web Development
Course, respond exactly:

"I can only answer questions related to the Sigma Web
Development Course."


IMPORTANT:

Only use information supported by the provided course context.

USER QUESTION:

{question}
"""


    return prompt


# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(

    "🔎 Ask a question",

    placeholder="Example: What is VS Code?"
)


# ============================================================
# SEARCH BUTTON
# ============================================================

if st.button("🔍 Search Course"):


    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # ========================================================
    # RETRIEVAL
    # ========================================================

    with st.spinner(
        "Searching the course..."
    ):

        try:

            results = search_course(
                question,
                TOP_K
            )

        except Exception as e:

            st.error(
                "Something went wrong while searching."
            )

            st.code(str(e))

            st.stop()


    # ========================================================
    # CREATE PROMPT
    # ========================================================

    prompt = create_prompt(
        question,
        results
    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    with st.spinner(
        "Generating answer..."
    ):

        try:

            answer = generate_answer(
                prompt
            )

        except Exception as e:

            st.error(
                "Could not generate the answer."
            )

            st.code(str(e))

            st.info(
                "Make sure your Ollama LLM model exists. "
                "Run 'ollama list' in PowerShell."
            )

            st.stop()


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    st.markdown(
        "## 🤖 Answer"
    )

    st.write(
        answer
    )


    # ========================================================
    # RELEVANT COURSE RESULTS
    # ========================================================

    st.markdown(
        "## 📚 Relevant Course Videos"
    )


    for rank, (_, row) in enumerate(

        results.iterrows(),

        start=1
    ):

        similarity = float(
            row["similarity"]
        )


        title = row.get(
            "title",
            "Unknown"
        )


        number = row.get(
            "number",
            "Unknown"
        )


        start = row.get(
            "start",
            0
        )


        end = row.get(
            "end",
            0
        )


        text = row.get(
            "text",
            ""
        )


        start_timestamp = format_timestamp(
            start
        )


        end_timestamp = format_timestamp(
            end
        )


        # ====================================================
        # RESULT CARD
        # ====================================================

        with st.expander(

            f"Result {rank} — "
            f"{title}"

        ):

            st.write(
                "**Video:**",
                title
            )


            st.write(
                "**Video Number:**",
                number
            )


            st.success(

                f"⏱️ Timestamp: "
                f"{start_timestamp} → "
                f"{end_timestamp}"

            )


            st.write(
                "**Similarity:**",
                round(
                    similarity,
                    4
                )
            )


            st.write(
                "**Course Content:**"
            )


            st.info(
                text
            )


    # ========================================================
    # SAVE PROMPT
    # ========================================================

    with open(
        "prompt.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(prompt)


    # ========================================================
    # SAVE RESPONSE
    # ========================================================

    with open(
        "response.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(answer)