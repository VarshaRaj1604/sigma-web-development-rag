import os
import json
import requests
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

JSON_FOLDER = "jsons"

OUTPUT_FILE = "embeddings.joblib"

OLLAMA_URL = "http://localhost:11434/api/embed"

EMBEDDING_MODEL = "nomic-embed-text"

# IMPORTANT:
# Do not send 1368 chunks in one request.
# Process them in smaller batches.
BATCH_SIZE = 32


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embedding(text_list):

    if not text_list:
        return []

    try:

        response = requests.post(

            OLLAMA_URL,

            json={
                "model": EMBEDDING_MODEL,
                "input": text_list
            },

            timeout=600
        )

    except requests.exceptions.ConnectionError:

        print("\nERROR: Ollama is not running.")

        print(
            "Start Ollama and try again."
        )

        raise


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    if response.status_code != 200:

        print("\nOLLAMA ERROR")

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
            "Ollama did not return embeddings."
        )


    return data["embeddings"]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "SIGMA WEB DEVELOPMENT"
    )

    print(
        "CREATING COURSE EMBEDDINGS"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Check JSON folder
    # --------------------------------------------------------

    if not os.path.exists(JSON_FOLDER):

        print(
            f"\nERROR: Folder '{JSON_FOLDER}' does not exist."
        )

        return


    json_files = sorted([

        file

        for file in os.listdir(JSON_FOLDER)

        if file.lower().endswith(".json")

    ])


    if not json_files:

        print(
            "\nNo JSON files found."
        )

        return


    print(
        "\nJSON files found:",
        len(json_files)
    )


    # --------------------------------------------------------
    # Store everything here
    # --------------------------------------------------------

    all_chunks = []

    chunk_id = 0


    # ========================================================
    # READ ALL JSON FILES
    # ========================================================

    for json_file in json_files:

        file_path = os.path.join(
            JSON_FOLDER,
            json_file
        )


        print("\n" + "-" * 70)

        print(
            "Processing:",
            json_file
        )


        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                content = json.load(f)

        except Exception as e:

            print(
                "Could not read:",
                e
            )

            continue


        # ----------------------------------------------------
        # Get chunks
        # ----------------------------------------------------

        chunks = content.get(
            "chunks",
            []
        )


        if not isinstance(chunks, list):

            print(
                "Skipping file because "
                "'chunks' is not a list."
            )

            continue


        print(
            "Number of chunks:",
            len(chunks)
        )


        # ----------------------------------------------------
        # Get metadata
        # ----------------------------------------------------

        file_title = content.get(
            "title",
            json_file
        )

        file_number = content.get(
            "number",
            ""
        )


        # ====================================================
        # PROCESS EACH CHUNK
        # ====================================================

        for chunk in chunks:

            if not isinstance(
                chunk,
                dict
            ):

                continue


            text = chunk.get(
                "text",
                ""
            )


            if not isinstance(
                text,
                str
            ):

                continue


            text = text.strip()


            if not text:

                continue


            # ------------------------------------------------
            # Create clean record
            # ------------------------------------------------

            new_chunk = {

                "chunk_id": chunk_id,

                "title": chunk.get(
                    "title",
                    file_title
                ),

                "number": chunk.get(
                    "number",
                    file_number
                ),

                "start": chunk.get(
                    "start",
                    chunk.get(
                        "start_time",
                        0
                    )
                ),

                "end": chunk.get(
                    "end",
                    chunk.get(
                        "end_time",
                        0
                    )
                ),

                "text": text,

                "source_file": json_file

            }


            all_chunks.append(
                new_chunk
            )


            chunk_id += 1


    # ========================================================
    # CHECK
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TOTAL CHUNKS:",
        len(all_chunks)
    )

    print("=" * 70)


    if not all_chunks:

        print(
            "\nNo valid chunks found."
        )

        return


    # ========================================================
    # CREATE EMBEDDINGS IN BATCHES
    # ========================================================

    print(
        "\nCreating embeddings..."
    )

    total = len(all_chunks)


    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )


        batch = all_chunks[
            start:end
        ]


        texts = [

            item["text"]

            for item in batch

        ]


        print(
            f"Embedding "
            f"{start + 1}-{end} "
            f"of {total}"
        )


        embeddings = create_embedding(
            texts
        )


        if len(embeddings) != len(
            batch
        ):

            raise RuntimeError(
                "Embedding count does not "
                "match chunk count."
            )


        for i in range(
            len(batch)
        ):

            batch[i][
                "embedding"
            ] = embeddings[i]


    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        all_chunks
    )


    # ========================================================
    # CREATE NUMPY MATRIX
    # ========================================================

    embedding_matrix = np.vstack(
        df["embedding"].values
    )


    print(
        "\nEmbedding matrix shape:",
        embedding_matrix.shape
    )


    # ========================================================
    # SAVE
    # ========================================================

    joblib.dump(
        df,
        OUTPUT_FILE
    )


    print(
        "\nEmbeddings successfully saved!"
    )

    print(
        "File:",
        OUTPUT_FILE
    )

    print(
        "Total chunks:",
        len(df)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()