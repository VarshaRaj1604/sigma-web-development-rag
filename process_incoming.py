import joblib
import requests
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_FILE = "embeddings.joblib"

OLLAMA_EMBED_URL = (
    "http://localhost:11434/api/embed"
)

OLLAMA_GENERATE_URL = (
    "http://localhost:11434/api/generate"
)

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "llama3.2"

TOP_K = 5

SIMILARITY_THRESHOLD = 0.40


# ============================================================
# CREATE QUESTION EMBEDDING
# ============================================================

def create_embedding(text):

    try:

        response = requests.post(

            OLLAMA_EMBED_URL,

            json={
                "model": EMBEDDING_MODEL,
                "input": [text]
            },

            timeout=300
        )

    except requests.exceptions.ConnectionError:

        print(
            "\nERROR: Ollama is not running."
        )

        raise


    if response.status_code != 200:

        print(
            "\nEmbedding Error"
        )

        print(
            "STATUS:",
            response.status_code
        )

        print(
            "BODY:",
            response.text
        )

        response.raise_for_status()


    data = response.json()


    if "embeddings" not in data:

        raise RuntimeError(
            "Embedding response is invalid."
        )


    return data["embeddings"][0]


# ============================================================
# GENERATE ANSWER USING OLLAMA
# ============================================================

def inference(prompt):

    try:

        response = requests.post(

            OLLAMA_GENERATE_URL,

            json={

                "model": LLM_MODEL,

                "prompt": prompt,

                "stream": False,

                "options": {

                    "temperature": 0.2

                }

            },

            timeout=600
        )

    except requests.exceptions.ConnectionError:

        print(
            "\nERROR: Ollama is not running."
        )

        raise


    if response.status_code != 200:

        print(
            "\nLLM Error"
        )

        print(
            "STATUS:",
            response.status_code
        )

        print(
            "BODY:",
            response.text
        )

        response.raise_for_status()


    data = response.json()


    return data["response"]


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print(
    "\nLoading embeddings..."
)


df = joblib.load(
    EMBEDDINGS_FILE
)


print(
    "Embeddings loaded successfully."
)


print(
    "Total chunks:",
    len(df)
)


# ============================================================
# CREATE EMBEDDING MATRIX ONCE
# ============================================================

print(
    "\nCreating embedding matrix..."
)


embedding_matrix = np.vstack(

    df[
        "embedding"
    ].values

)


print(
    "Matrix shape:",
    embedding_matrix.shape
)


# ============================================================
# SEARCH COURSE
# ============================================================

def search_course(
    question,
    top_k=TOP_K
):

    # --------------------------------------------------------
    # Question embedding
    # --------------------------------------------------------

    question_embedding = (
        create_embedding(
            question
        )
    )


    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(

        embedding_matrix,

        [question_embedding]

    ).flatten()


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    top_indices = (

        similarities.argsort()[::-1]

        [:top_k]

    )


    results = []


    # --------------------------------------------------------
    # Add results
    # --------------------------------------------------------

    for index in top_indices:

        score = float(
            similarities[index]
        )


        row = df.iloc[
            index
        ]


        results.append({

            "index": int(index),

            "score": score,

            "title": row.get(
                "title",
                "Unknown"
            ),

            "number": row.get(
                "number",
                "Unknown"
            ),

            "start": row.get(
                "start",
                0
            ),

            "end": row.get(
                "end",
                0
            ),

            "text": row.get(
                "text",
                ""
            )

        })


    return results


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(seconds):

    try:

        seconds = float(
            seconds
        )

    except:

        return "Unknown"


    minutes = int(
        seconds // 60
    )

    seconds = int(
        seconds % 60
    )


    return (
        f"{minutes:02d}:{seconds:02d}"
    )


# ============================================================
# BUILD COURSE CONTEXT
# ============================================================

def build_context(
    results
):

    context = []


    for result in results:

        start_time = format_time(
            result["start"]
        )

        end_time = format_time(
            result["end"]
        )


        context.append({

            "video_title":
                result["title"],

            "video_number":
                result["number"],

            "start_time":
                result["start"],

            "end_time":
                result["end"],

            "text":
                result["text"],

            "similarity":
                round(
                    result["score"],
                    4
                )

        })


    return context


# ============================================================
# BUILD PROMPT
# ============================================================

def create_prompt(
    question,
    context
):

    prompt = f"""
You are an AI assistant for the
Sigma Web Development Course.

Your job is to help students find
information inside the course.

USER QUESTION:

{question}


COURSE CONTEXT:

{context}


INSTRUCTIONS:

Answer the user's question using ONLY
the course context above.

If the question is related to the course:

1. Give a clear and simple answer.

2. Explain what is being taught.

3. Tell the user the relevant video.

4. Give the video number if available.

5. Give the approximate timestamp.

6. Tell the student which video they
   should watch.

7. If several results are relevant,
   mention the most relevant one first.

8. Use ONLY the timestamps provided.

9. NEVER invent timestamps.

10. NEVER invent video titles.

11. NEVER invent video numbers.

12. Do not mention embeddings.

13. Do not mention cosine similarity.

14. Do not mention chunks.

15. Do not mention JSON.

16. Do not mention the prompt.

17. Do not mention retrieval.

18. Answer naturally like a helpful
    course assistant.


If the question is NOT related to
the Sigma Web Development Course,
respond exactly:

"I can only answer questions related
to the Sigma Web Development Course."


If the course context does not contain
enough information to answer the question,
say:

"I could not find enough information
about this topic in the available
course material."
"""


    return prompt


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_results(
    results
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RELEVANT COURSE SECTIONS"
    )

    print(
        "=" * 70
    )


    for rank, result in enumerate(

        results,

        start=1

    ):

        print(
            "\n" + "-" * 70
        )


        print(
            f"Result {rank}"
        )


        print(
            "Similarity:",
            round(
                result["score"],
                4
            )
        )


        print(
            "Video:",
            result["title"]
        )


        print(
            "Video Number:",
            result["number"]
        )


        print(
            "Timestamp:",
            format_time(
                result["start"]
            ),
            "-",
            format_time(
                result["end"]
            )
        )


        print(
            "Text:",
            result["text"]
        )


# ============================================================
# MAIN QUESTION LOOP
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SIGMA WEB DEVELOPMENT"
)

print(
    "RAG COURSE ASSISTANT"
)

print(
    "=" * 70
)

print(
    "\nType 'exit' to quit."
)


while True:

    print(
        "\n" + "-" * 70
    )


    incoming_query = input(
        "Ask a Question: "
    ).strip()


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if incoming_query.lower() == "exit":

        print(
            "\nGoodbye!"
        )

        break


    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    if not incoming_query:

        print(
            "Please enter a question."
        )

        continue


    try:

        print(
            "\nSearching course..."
        )


        # ====================================================
        # SEARCH
        # ====================================================

        results = search_course(
            incoming_query
        )


        # ====================================================
        # CHECK SIMILARITY
        # ====================================================

        best_score = results[0][
            "score"
        ]


        print(
            "Best similarity:",
            round(
                best_score,
                4
            )
        )


        if best_score < (
            SIMILARITY_THRESHOLD
        ):

            print(
                "\nI could not find "
                "relevant information "
                "in the available "
                "course material."
            )

            continue


        # ====================================================
        # DISPLAY
        # ====================================================

        display_results(
            results
        )


        # ====================================================
        # CONTEXT
        # ====================================================

        context = build_context(
            results
        )


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = create_prompt(

            incoming_query,

            context

        )


        # ----------------------------------------------------
        # Save prompt
        # ----------------------------------------------------

        with open(

            "prompt.txt",

            "w",

            encoding="utf-8"

        ) as f:

            f.write(prompt)


        # ====================================================
        # LLM
        # ====================================================

        print(
            "\nGenerating answer..."
        )


        answer = inference(
            prompt
        )


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ANSWER"
        )

        print(
            "=" * 70
        )

        print(
            answer
        )


        # ====================================================
        # SAVE ANSWER
        # ====================================================

        with open(

            "response.txt",

            "w",

            encoding="utf-8"

        ) as f:

            f.write(answer)


    except Exception as e:

        print(
            "\nERROR:"
        )

        print(
            str(e)
        )