"""Notebook creation and reopening endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.persistence.errors import PersistenceError
from core.persistence.models import new_id
from core.persistence.store import PersistenceStore

router = APIRouter()


class NotebookCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    notebook_id: str | None = None


class NotebookPayload(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class SourcePayload(BaseModel):
    id: str
    notebook_id: str
    display_name: str
    created_at: str
    updated_at: str
    current_import_id: str | None
    pending_import_id: str | None


def _store(request: Request) -> PersistenceStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="notebook store is unavailable")
    return store


@router.get("/notebooks", response_model=list[NotebookPayload])
def list_notebooks(request: Request) -> list[NotebookPayload]:
    return [
        NotebookPayload(
            id=item.id,
            title=item.title,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in _store(request).list_notebooks()
    ]


@router.post("/notebooks", response_model=NotebookPayload, status_code=201)
def create_notebook(payload: NotebookCreate, request: Request) -> NotebookPayload:
    try:
        notebook = _store(request).create_notebook(payload.notebook_id or new_id(), payload.title)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return NotebookPayload(
        id=notebook.id,
        title=notebook.title,
        created_at=notebook.created_at,
        updated_at=notebook.updated_at,
    )


@router.get("/notebooks/{notebook_id}", response_model=NotebookPayload)
def get_notebook(notebook_id: str, request: Request) -> NotebookPayload:
    try:
        notebook = _store(request).load_notebook(notebook_id)
    except PersistenceError as error:
        raise HTTPException(status_code=404, detail="notebook does not exist") from error
    return NotebookPayload(
        id=notebook.id,
        title=notebook.title,
        created_at=notebook.created_at,
        updated_at=notebook.updated_at,
    )


@router.get("/notebooks/{notebook_id}/sources", response_model=list[SourcePayload])
def list_sources(notebook_id: str, request: Request) -> list[SourcePayload]:
    store = _store(request)
    try:
        store.load_notebook(notebook_id)
    except (PersistenceError, ValueError) as error:
        raise HTTPException(status_code=404, detail="notebook does not exist") from error
    result: list[SourcePayload] = []
    for source, state in store.list_notebook_sources(notebook_id):
        result.append(
            SourcePayload(
                id=source.id,
                notebook_id=source.notebook_id,
                display_name=source.display_name,
                created_at=source.created_at,
                updated_at=source.updated_at,
                current_import_id=None if state is None else state.current_import_id,
                pending_import_id=None if state is None else state.pending_import_id,
            )
        )
    return result
