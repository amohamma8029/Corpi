"""Finite embedding protocol limits and safe errors shared by parent and child."""

MAX_MESSAGE_BYTES = 2 * 1024 * 1024

# Never propagate arbitrary native text, even in a field labelled "code".
WORKER_ERROR_CODES = frozenset(
    {
        "invalid_embedding_input",
        "invalid_embedding_batch",
        "invalid_vector",
        "invalid_vector_count",
        "model_input_too_long",
        "model_corrupt",
        "model_not_installed",
        "runtime_mismatch",
        "tokenizer_failed",
        "embedding_message_limit",
        "embedding_worker_failed",
    }
)
