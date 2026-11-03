"""Explicit meaning/word queries behind the existing local authenticated boundary."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.persistence.errors import ConflictingWriteError, PersistenceError
from core.search.query import QueryService, SearchFailure, SearchMode

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8192)
    mode: SearchMode = "meaning"
    limit: int = Field(default=10, ge=1, le=50)


class SearchCitation(BaseModel):
    passage_id: str
    ordinal: int
    start_offset: int | None
    end_offset: int | None
    page_number: int | None


class SearchResult(BaseModel):
    source_id: str
    source_version_id: str
    index_version_id: str
    chunk_id: str
    manifest_id: str
    source_name: str
    text: str
    score: float
    import_id: str | None
    filename: str | None
    profile_id: str | None
    pdf_role: Literal["current", "previous"] | None
    citations: list[SearchCitation]


class SearchResponse(BaseModel):
    notebook_id: str
    mode: SearchMode
    revision: str
    ready_sources: int
    results: list[SearchResult]


class SearchState(BaseModel):
    notebook_id: str
    revision: str


@router.get("/notebooks/{notebook_id}/search-state", response_model=SearchState)
def search_state(notebook_id: str, request: Request) -> SearchState:
    service: QueryService | None = getattr(request.app.state, "search_queries", None)
    if service is None:
        raise HTTPException(503, "local search is unavailable")
    try:
        return SearchState.model_validate(service.state(notebook_id))
    except (PersistenceError, ValueError):
        raise HTTPException(404, "notebook or search data is unavailable") from None


@router.post("/notebooks/{notebook_id}/search", response_model=SearchResponse)
def search_notebook(notebook_id: str, payload: SearchRequest, request: Request) -> SearchResponse:
    service: QueryService | None = getattr(request.app.state, "search_queries", None)
    if service is None:
        raise HTTPException(503, "local search is unavailable")
    try:
        return SearchResponse.model_validate(
            service.search(notebook_id, payload.query, payload.mode, limit=payload.limit)
        )
    except SearchFailure as error:
        raise HTTPException(error.status, {"code": error.code, "message": str(error)}) from None
    except ConflictingWriteError:
        raise HTTPException(
            409,
            {
                "code": "search_input_changed",
                "message": "Search data changed. Please search again.",
            },
        ) from None
    except PersistenceError:
        raise HTTPException(404, "notebook or search data is unavailable") from None
    except ValueError:
        raise HTTPException(
            400,
            {"code": "invalid_search_query", "message": "Enter a shorter, non-empty search query."},
        ) from None
