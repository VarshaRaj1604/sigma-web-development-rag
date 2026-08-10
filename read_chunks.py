import requests
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SETTINGS
# ============================================================

JSON_FOLDER = "jsons"

# Keep this small because Ollama was previously giving
# /tokenize connection refused errors.
BATCH_SIZE = 4

# Number of results to show for the question
TOP_RESULTS = 3

# Ollama embedding model
MODEL_NAME = "nomic-embed-text"


# ============================================================
# Create embeddings using Ollama
# ============================================================

def create_embeddings(text_list, batch_size=BATCH_SIZE):

    url = "http://localhost:11434/api/embed"

    all_embeddings = []

    total_texts = len(text_list)

    print(
        f"Creating embeddings for {total_texts} chunks..."
    )

    # Process chunks in batches
    for start in range(
        0,
        total_texts,
        batch_size
    ):

        end = min(
            start + batch_size,
            total_texts
        )

        batch = text_list[start:end]

        print(
            f"Embedding chunks "
            f"{start + 1}-{end} "
            f"of {total_texts}"
        )

        try:

            response = requests.post(
                url,
                json={
                    "model": MODEL_NAME,
                    "input": batch
                },
                timeout=300
            )

        except requests.exceptions.ConnectionError:

            print("\nERROR: Cannot connect to Ollama.")

            print(
                "Make sure Ollama is running:"
            )

            print(
                "ollama serve"
            )

            raise


        # Check response
        if response.status_code != 200:

            print("\nOllama Error")

            print(
                "STATUS:",
                response.status_code
            )

            print(
                "BODY:",
                response.text
            )

            response.raise_for_status()


        # Get embeddings
        batch_embeddings = response.json()["embeddings"]

        # Add them to the complete list
        all_embeddings.extend(
            batch_embeddings
        )


    print(
        f"Successfully created "
        f"{len(all_embeddings)} embeddings."
    )

    return all_embeddings


# ============================================================
# Check JSON folder
# ============================================================

if not os.path.exists(JSON_FOLDER):

    print(
        f"ERROR: '{JSON_FOLDER}' folder does not exist."
    )

    print(
        "Make sure the jsons folder is in the same "
        "folder as read_chunks.py."
    )

    exit()


# ============================================================
# Get JSON files
# ============================================================

json_files = [
    file
    for file in os.listdir(JSON_FOLDER)
    if file.endswith(".json")
]


print(
    f"Found {len(json_files)} JSON files."
)


# ============================================================
# Store all chunks
# ============================================================

my_dicts = []

chunk_id = 0


# ============================================================
# Process every JSON file
# ============================================================

for json_file in json_files:

    print("\n" + "=" * 70)

    print(
        f"Processing: {json_file}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # File path
    # --------------------------------------------------------

    file_path = os.path.join(
        JSON_FOLDER,
        json_file
    )


    # --------------------------------------------------------
    # Read JSON
    # --------------------------------------------------------

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        content = json.load(f)


    # --------------------------------------------------------
    # Get chunks
    # --------------------------------------------------------

    chunks = content.get(
        "chunks",
        []
    )


    print(
        "Total chunks in file:",
        len(chunks)
    )


    if not chunks:

        print(
            "No chunks found. Skipping..."
        )

        continue


    # --------------------------------------------------------
    # Prepare valid chunks
    # --------------------------------------------------------

    valid_chunks = []

    for chunk in chunks:

        # Make sure chunk is a dictionary
        if not isinstance(chunk, dict):

            continue


        # Get text
        text = chunk.get(
            "text",
            ""
        )


        if text is None:

            continue


        text = str(text).strip()


        # Skip empty text
        if not text:

            continue


        # Store normalized chunk
        valid_chunks.append(
            {
                **chunk,
                "text": text
            }
        )


    print(
        "Valid chunks:",
        len(valid_chunks)
    )


    if not valid_chunks:

        print(
            "No valid text chunks. Skipping..."
        )

        continue


    # --------------------------------------------------------
    # Get text from ALL chunks
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in valid_chunks
    ]


    # --------------------------------------------------------
    # Create embeddings for ALL chunks
    # --------------------------------------------------------

    embeddings = create_embeddings(
        texts,
        batch_size=BATCH_SIZE
    )


    # --------------------------------------------------------
    # Check embedding count
    # --------------------------------------------------------

    if len(embeddings) != len(valid_chunks):

        raise ValueError(
            f"Embedding count mismatch! "
            f"Chunks={len(valid_chunks)}, "
            f"Embeddings={len(embeddings)}"
        )


    # --------------------------------------------------------
    # Add embeddings to chunks
    # --------------------------------------------------------

    for i, chunk in enumerate(
        valid_chunks
    ):

        chunk["chunk_id"] = chunk_id

        chunk["embedding"] = embeddings[i]

        # Save source filename
        chunk["source_file"] = json_file

        my_dicts.append(
            chunk
        )

        chunk_id += 1


    print(
        f"Finished: {json_file}"
    )


# ============================================================
# Create DataFrame
# ============================================================

print("\n" + "=" * 70)

print(
    "Creating DataFrame..."
)

print("=" * 70)


if not my_dicts:

    print(
        "ERROR: No chunks were processed."
    )

    exit()


df = pd.DataFrame.from_records(
    my_dicts
)


print(
    "\nDataFrame created successfully!"
)

print(
    "Total chunks:",
    len(df)
)

print(
    "Columns:",
    list(df.columns)
)


# ============================================================
# Create COMPLETE embedding matrix
# ============================================================

print("\nCreating embedding matrix...")


embedding_matrix = np.vstack(
    df["embedding"].values
)


print(
    "Embedding matrix shape:",
    embedding_matrix.shape
)


# Example:
#
# (5000, 768)
#
# 5000 = total chunks
# 768  = embedding dimensions
#


# ============================================================
# Save DataFrame
# ============================================================

joblib.dump(
    df,
    "embeddings.joblib"
)


print(
    "\nEmbeddings saved to:"
)

print(
    "embeddings.joblib"
)


# ============================================================
# Ask a question
# ============================================================

incoming_query = input(
    "\nAsk a Question: "
)


# ============================================================
# Create embedding for question
# ============================================================

question_embedding = create_embeddings(
    [incoming_query],
    batch_size=1
)[0]


print(
    "\nQuestion embedding created."
)


# ============================================================
# Calculate cosine similarity
# ============================================================

print(
    "\nCalculating cosine similarity..."
)


similarities = cosine_similarity(
    embedding_matrix,
    [question_embedding]
).flatten()


print(
    "Similarity calculation completed."
)


# ============================================================
# Get top results
# ============================================================

top_indices = similarities.argsort()[
    ::-1
][:TOP_RESULTS]


# ============================================================
# Display similarity scores
# ============================================================

print("\n" + "=" * 70)

print(
    "TOP SIMILARITY SCORES"
)

print("=" * 70)


for rank, index in enumerate(
    top_indices,
    start=1
):

    print(
        f"{rank}. "
        f"Chunk {df.iloc[index]['chunk_id']} "
        f"-> "
        f"{similarities[index]:.4f}"
    )


# ============================================================
# Display relevant chunks
# ============================================================

print("\n" + "=" * 70)

print(
    "MOST RELEVANT RESULTS"
)

print("=" * 70)


for rank, index in enumerate(
    top_indices,
    start=1
):

    row = df.iloc[index]


    print(
        "\n" + "-" * 70
    )


    print(
        f"RESULT {rank}"
    )


    print(
        "-" * 70
    )


    print(
        "Similarity:",
        round(
            similarities[index],
            4
        )
    )


    print(
        "Chunk ID:",
        row.get(
            "chunk_id",
            "Not available"
        )
    )


    print(
        "Source:",
        row.get(
            "source_file",
            "Not available"
        )
    )


    print(
        "Title:",
        row.get(
            "title",
            "Not available"
        )
    )


    print(
        "Number:",
        row.get(
            "number",
            "Not available"
        )
    )


    print(
        "\nText:"
    )


    print(
        row.get(
            "text",
            "No text available"
        )
    )


print(
    "\nFinished!"
)