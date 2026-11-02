"""Local-only FastEmbed CPU adapter with explicit, truncation-free input checks."""

from __future__ import annotations

import math
import os
import struct
from collections import OrderedDict
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from core.search.profiles import ModelProfile, verify_installation

if TYPE_CHECKING:
    from tokenizers import Tokenizer


def masked_mean_pool(embeddings, attention_mask):
    """Same masked mean with float64 accumulation, without tiled 3-D buffers."""
    import numpy as np

    mask = np.asarray(attention_mask, dtype=np.float64)
    sums = np.einsum("bsh,bs->bh", embeddings, mask, dtype=np.float64, optimize=False)
    counts = np.maximum(mask.sum(axis=1, keepdims=True), 1e-9)
    return sums / counts


def vector_bytes(values: Sequence[float], dimensions: int) -> bytes:
    if len(values) != dimensions or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values
    ):
        raise ValueError("invalid_vector")
    norm = math.hypot(*values)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("invalid_vector")
    result = struct.pack(f"<{dimensions}f", *(v / norm for v in values))
    unpacked = struct.unpack(f"<{dimensions}f", result)
    if not all(math.isfinite(v) for v in unpacked) or math.hypot(*unpacked) == 0:
        raise ValueError("invalid_vector")
    return result


class InputTokenizer:
    # Existing benchmark tokenizers override __init__ and keep unpadded counts.
    _preserve_padding = False

    def __init__(
        self, path: Path, profile: ModelProfile, *, tokenizer: Tokenizer | None = None
    ) -> None:
        self.profile = profile
        self._preserve_padding = tokenizer is not None
        try:
            if tokenizer is None:
                from fastembed.common.preprocessor_utils import load_tokenizer

                tokenizer, _ = load_tokenizer(path)
            self.tokenizer = tokenizer
            self.tokenizer.no_truncation()
            if not self._preserve_padding:
                self.tokenizer.no_padding()
        except Exception:
            # Planning runs in the parent too: native diagnostics must never
            # become durable job errors (which are exposed through the API).
            raise ValueError("tokenizer_failed") from None
        # Counts are deterministic for this tokenizer and its fixed settings.
        # Keep a small instance-local cache; large source strings are not retained.
        self._counts: OrderedDict[tuple[str, bool], int] = OrderedDict()

    def _encode_count(self, text: str, special: bool) -> int:
        try:
            encoded = self.tokenizer.encode(text, add_special_tokens=special)
            # The runtime shares its inference tokenizer. Count real tokens without
            # changing batch padding, including fixed/multiple and left padding.
            return sum(encoded.attention_mask) if self._preserve_padding else len(encoded.ids)
        except Exception:
            raise ValueError("tokenizer_failed") from None

    def _count(self, text: str, special: bool) -> int:
        if len(text) > 16_384:
            return self._encode_count(text, special)
        key = (text, special)
        count = self._counts.pop(key, None)
        if count is None:
            count = self._encode_count(text, special)
        self._counts[key] = count
        if len(self._counts) > 128:
            self._counts.popitem(last=False)
        return count

    def count(self, text: str) -> int:
        return self._count(text, False)

    def prepare(self, text: str, *, query: bool = False) -> str:
        prepared = (self.profile.query_prefix if query else self.profile.passage_prefix) + text
        if self._count(prepared, True) > self.profile.input_tokens:
            raise ValueError("model_input_too_long")
        return prepared


def _load_runtime_tokenizer(path: Path):
    """Parse before backend imports; retain FastEmbed 0.8.1's exact configuration.

    Its text loader parses after session allocation. The runtime instead separates
    that large parse from both backend imports and session allocation. Configuration
    below follows the pinned loader and is checked against its serialized tokenizer.
    """
    import json

    from tokenizers import AddedToken, Tokenizer

    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    from fastembed.common.preprocessor_utils import (
        _resolve_max_context,
        iter_special_tokens,
        load_special_tokens,
    )

    config_path = path / "config.json"
    config = json.loads(config_path.read_text("utf-8")) if config_path.exists() else {}
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text("utf-8"))
    tokenizer.enable_truncation(max_length=_resolve_max_context(tokenizer_config, path))
    for token in iter_special_tokens(load_special_tokens(path)):
        if isinstance(token, str):
            tokenizer.add_special_tokens([token])
        elif isinstance(token, dict):
            tokenizer.add_special_tokens([AddedToken(**token)])
    padding = tokenizer.padding or {}
    pad_token = padding.get("pad_token") or tokenizer_config.get("pad_token")
    if pad_token is None:
        raise ValueError("tokenizer_failed")
    pad_id = padding.get("pad_id", config.get("pad_token_id"))
    if pad_id is None:
        pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise ValueError("tokenizer_failed")
    tokenizer.enable_padding(
        direction=padding.get("direction", "right"),
        pad_id=pad_id,
        pad_type_id=padding.get("pad_type_id", 0),
        pad_token=pad_token,
        pad_to_multiple_of=padding.get("pad_to_multiple_of"),
        length=None,
    )
    special_token_to_id = {
        token.content: token_id
        for token_id, token in tokenizer.get_added_tokens_decoder().items()
        if token.special
    }
    return tokenizer, special_token_to_id


class EmbeddingRuntime:
    def __init__(
        self,
        path: Path,
        profile: ModelProfile,
        *,
        threads: int = 2,
        memory_arena: bool = True,
        stop_spinning: bool = True,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        verify_installation(path, profile)
        if version("fastembed") != "0.8.1":
            raise ValueError("runtime_mismatch")
        self.profile = profile
        try:
            native_tokenizer, special_token_to_id = _load_runtime_tokenizer(path)
        except Exception:
            raise ValueError("tokenizer_failed") from None
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        from fastembed.common.onnx_model import OnnxModel
        from fastembed.text.custom_text_embedding import CustomTextEmbedding
        from tokenizers import Tokenizer

        class RestingCustomEmbedding(CustomTextEmbedding):
            def _load_onnx_model(self, *args, **kwargs):
                # In pinned FastEmbed 0.8.1 the text loader only calls this base
                # session loader, then loads a tokenizer. Keep that session path
                # unchanged while reusing the tokenizer loaded before the session.
                OnnxModel._load_onnx_model(self, *args, **kwargs)
                self.tokenizer = native_tokenizer
                self.special_token_to_id = special_token_to_id

            @classmethod
            def add_extra_session_options(cls, session_options, extra_options):
                super().add_extra_session_options(session_options, extra_options)
                if stop_spinning:
                    session_options.add_session_config_entry("session.force_spinning_stop", "1")

        class RestingTextEmbedding(TextEmbedding):
            EMBEDDINGS_REGISTRY = [RestingCustomEmbedding]

        # Register a Corpi-owned name so pooling/prefixes are explicit for every
        # artifact, including E5, rather than relying on a mutable built-in list.
        name = "corpi/" + profile.identity
        supported = TextEmbedding.list_supported_models()
        if not any(item["model"] == name for item in supported):
            TextEmbedding.add_custom_model(
                model=name,
                pooling=PoolingType.CLS if profile.pooling == "cls" else PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=profile.repository),
                dim=profile.dimensions,
                model_file=profile.model_file,
            )
        self.model = RestingTextEmbedding(
            model_name=name,
            specific_model_path=str(path),
            cache_dir=str(path),
            local_files_only=True,
            threads=max(1, min(threads, 2, os.cpu_count() or 1)),
            providers=["CPUExecutionProvider"],
            cuda=False,
            enable_cpu_mem_arena=memory_arena,
        )
        # Parsing the large tokenizer JSON precedes session allocation. Backend
        # and validation retain one vocabulary without changing inference padding.
        backend_tokenizer = getattr(self.model.model, "tokenizer", None)
        if (
            not isinstance(backend_tokenizer, Tokenizer)
            or backend_tokenizer is not native_tokenizer
        ):
            raise ValueError("tokenizer_failed")
        self.tokenizer = InputTokenizer(path, profile, tokenizer=backend_tokenizer)

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[bytes]:
        if not 1 <= len(texts) <= 32:
            raise ValueError("invalid_embedding_batch")
        prepared = [self.tokenizer.prepare(text, query=query) for text in texts]
        values = list(self.model.embed(prepared, batch_size=len(texts), parallel=None))
        if len(values) != len(texts):
            raise ValueError("invalid_vector_count")
        return [vector_bytes(value.tolist(), self.profile.dimensions) for value in values]
