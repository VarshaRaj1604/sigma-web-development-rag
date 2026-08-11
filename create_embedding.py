import os
import json
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

JSON_FOLDER = "jsons"
OUTPUT_FILE = "embeddings.joblib"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded!")


# ============================================================
# READ JSON FILES
# ============================================================

all_chunks = []

json_files = [
    file
    for file in os.listdir(JSON_FOLDER)
    if file.endswith(".json")
]

print("Total JSON files:", len(json_files))

chunk_id = 0


# ============================================================
# PROCESS ALL JSON FILES
# ============================================================

for json_file in json_files:

    file_path = os.path.join(
        JSON_FOLDER,
        json_file
    )

    print("\n========================================")
    print("Processing:", json_file)
    print("========================================")

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    chunks = data.get("chunks", [])

    print("Total chunks:", len(chunks))

    texts = []
    valid_chunks = []

    # --------------------------------------------------------
    # GET VALID CHUNKS
    # --------------------------------------------------------

    for chunk in chunks:

        text = chunk.get("text", "").strip()

        if not text:
            continue

        texts.append(text)
        valid_chunks.append(chunk)

    if not texts:
        print("No valid text found.")
        continue

    print(
        "Creating embeddings for:",
        len(texts),
        "chunks"
    )

    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
   
    # --------------------------------------------------------
    # STORE CHUNKS
    # --------------------------------------------------------

    for i, chunk in enumerate(valid_chunks):

        # ====================================================
        # GET TIMESTAMP
        # ====================================================

        start_time = chunk.get(
            "start",
            chunk.get(
                "start_time",
                0
            )
        )

        end_time = chunk.get(
            "end",
            chunk.get(
                "end_time",
                0
            )
        )

        # ====================================================
        # CREATE RECORD
        # ====================================================

        record = {

            "chunk_id": chunk_id,

            # Video title
            "title": chunk.get(
                "title",
                "Unknown"
            ),

            # Video number
            "number": chunk.get(
                "number",
                "Unknown"
            ),

            # Timestamp
            "start": start_time,

            "end": end_time,

            # Subtitle text
            "text": text,

            # Embedding
            "embedding": embeddings[i].tolist()
        }

        all_chunks.append(record)

        chunk_id += 1


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(all_chunks)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n========================================")
print("EMBEDDING CREATION COMPLETED")
print("========================================")

print(
    "Total chunks:",
    len(df)
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# CHECK TIMESTAMPS
# ============================================================

print(
    "\nSample timestamp data:"
)

if len(df) > 0:

    print(
        df[
            [
                "chunk_id",
                "title",
                "number",
                "start",
                "end",
                "text"
            ]
        ].head(10).to_string(
            index=False
        )
    )


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

joblib.dump(
    df,
    OUTPUT_FILE
)

print(
    f"\nSaved embeddings to: {OUTPUT_FILE}"
)

print(
    "\nDONE!"
)