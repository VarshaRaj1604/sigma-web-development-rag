# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing your JSON subtitle files
JSON_FOLDER = "jsons"

# Where embeddings will be saved
EMBEDDINGS_FILE = "data/embeddings.joblib"


# ============================================================
# OLLAMA
# ============================================================

OLLAMA_URL = "http://localhost:11434"

# Embedding model
EMBEDDING_MODEL = "nomic-embed-text"

# Local LLM
LLM_MODEL = "llama3.2"


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

# Number of best chunks to retrieve
TOP_K = 5

# Number of neighboring chunks
NEIGHBOR_COUNT = 1

# Minimum similarity score
SIMILARITY_THRESHOLD = 0.40


# ============================================================
# EMBEDDING SETTINGS
# ============================================================

# Number of texts sent to Ollama at once
BATCH_SIZE = 32


# ============================================================
# CHUNK SETTINGS
# ============================================================

MIN_CHUNK_SIZE = 300

MAX_CHUNK_SIZE = 800

OVERLAP_SIZE = 100