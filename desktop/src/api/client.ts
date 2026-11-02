import type { components, paths } from "./generated/schema";

export interface CoreConnection {
  base_url: string;
  token: string;
}

export interface CoreApiErrorDetail {
  code?: string;
  message?: string;
}

export class CoreApiError extends Error {
  readonly status: number;
  readonly code: string | undefined;

  constructor(path: string, status: number, detail: unknown) {
    const parsed = errorDetail(detail);
    super(parsed.message ?? `${path}: HTTP ${status}`);
    this.name = "CoreApiError";
    this.status = status;
    this.code = parsed.code;
  }
}

export type Notebook = components["schemas"]["NotebookPayload"];
export type NotebookSource = components["schemas"]["SourcePayload"];
export type PdfAcquisition = components["schemas"]["AcquisitionPayload"];
export type PdfImport = components["schemas"]["ImportPayload"];
export type StorageReport = components["schemas"]["StoragePayload"];
export type PdfImportTrash = components["schemas"]["TrashPayload"];
export type ProcessingAction = components["schemas"]["ProcessingActionPayload"];
export type ExtractionPageReport = components["schemas"]["ExtractionPagePayload"];
export type ExtractionReview = components["schemas"]["ExtractionReviewPayload"];
export type ExtractionReport = components["schemas"]["ExtractionReportPayload"];
export type SupportedModel = components["schemas"]["ModelSetupPayload"];
export type NotebookModel = components["schemas"]["NotebookModelPayload"];
export type NotebookSearchResult = components["schemas"]["SearchResult"];
export type NotebookSearchResponse = components["schemas"]["SearchResponse"];

export function searchNotebook(
  connection: CoreConnection, notebookId: string, query: string,
  mode: "meaning" | "words", signal?: AbortSignal,
): Promise<NotebookSearchResponse> {
  return coreRequest(connection, `/notebooks/${encodeURIComponent(notebookId)}/search`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode, limit: 10 }), signal,
  });
}

export function listModels(connection: CoreConnection): Promise<SupportedModel[]> {
  return coreRequest(connection, "/models");
}

export function getNotebookModel(connection: CoreConnection, notebookId: string): Promise<NotebookModel> {
  return coreRequest(connection, `/notebooks/${encodeURIComponent(notebookId)}/model`);
}

export function setupModel(connection: CoreConnection, model: SupportedModel): Promise<SupportedModel> {
  return coreRequest(connection, `/models/${encodeURIComponent(model.profile_id)}/setup`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_key: crypto.randomUUID(), generation: model.generation }),
  });
}

export function cancelModelSetup(connection: CoreConnection, model: SupportedModel): Promise<SupportedModel> {
  return coreRequest(connection, `/models/${encodeURIComponent(model.profile_id)}/setup/cancel`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generation: model.generation }),
  });
}

export function chooseNotebookModel(connection: CoreConnection, notebook: NotebookModel, profileId: string): Promise<NotebookModel> {
  return coreRequest(connection, `/notebooks/${encodeURIComponent(notebook.notebook_id)}/model`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_id: profileId, request_key: crypto.randomUUID(), generation: notebook.generation }),
  });
}

export function cancelNotebookModel(connection: CoreConnection, notebook: NotebookModel): Promise<NotebookModel> {
  return coreRequest(connection, `/notebooks/${encodeURIComponent(notebook.notebook_id)}/model/cancel`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generation: notebook.generation }),
  });
}
export type CoreHealth = paths["/health"]["get"]["responses"][200]["content"]["application/json"];

export async function coreRequest<T>(
  connection: CoreConnection,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Authorization", `Bearer ${connection.token}`);
  const response = await fetch(`${connection.base_url}${path}`, { ...init, headers });
  if (!response.ok) {
    throw new CoreApiError(path, response.status, await response.json().catch(() => undefined));
  }
  return (await response.json()) as T;
}

export async function coreBinary(
  connection: CoreConnection,
  path: string,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const headers = new Headers();
  headers.set("Authorization", `Bearer ${connection.token}`);
  const response = await fetch(`${connection.base_url}${path}`, { headers, signal });
  if (!response.ok) {
    throw new CoreApiError(path, response.status, await response.json().catch(() => undefined));
  }
  const buffer = await response.arrayBuffer();
  return new Uint8Array(buffer);
}

export function listNotebooks(connection: CoreConnection): Promise<Notebook[]> {
  return coreRequest(connection, "/notebooks");
}

export function createNotebook(connection: CoreConnection, title: string): Promise<Notebook> {
  return coreRequest(connection, "/notebooks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export function listNotebookSources(
  connection: CoreConnection,
  notebookId: string,
): Promise<NotebookSource[]> {
  return coreRequest(connection, `/notebooks/${encodeURIComponent(notebookId)}/sources`);
}

export function createPdfAcquisition(
  connection: CoreConnection,
  notebookId: string,
  payload: {
    retry_key: string;
    filename: string;
    intent: "add" | "replace";
    target_source_id?: string;
  },
): Promise<PdfAcquisition> {
  return coreRequest(connection, `/notebooks/${encodeURIComponent(notebookId)}/pdf-acquisitions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function listPdfAcquisitions(
  connection: CoreConnection,
  notebookId: string,
): Promise<PdfAcquisition[]> {
  return coreRequest(connection, `/notebooks/${encodeURIComponent(notebookId)}/pdf-acquisitions`);
}

export async function uploadPdf(
  connection: CoreConnection,
  acquisitionId: string,
  file: File,
  signal?: AbortSignal,
): Promise<PdfAcquisition> {
  const headers = new Headers({ "Content-Type": "application/pdf" });
  headers.set("Authorization", `Bearer ${connection.token}`);
  const path = `/pdf-acquisitions/${encodeURIComponent(acquisitionId)}/content`;
  const response = await fetch(`${connection.base_url}${path}`, {
    method: "PUT",
    headers,
    body: file,
    signal,
  });
  const payload = await response.json().catch(() => undefined);
  if (!response.ok) throw new CoreApiError(path, response.status, payload);
  return payload as PdfAcquisition;
}

export function getPdfAcquisition(
  connection: CoreConnection,
  acquisitionId: string,
): Promise<PdfAcquisition> {
  return coreRequest(connection, `/pdf-acquisitions/${encodeURIComponent(acquisitionId)}`);
}

export function retryPdfAcquisition(
  connection: CoreConnection,
  acquisitionId: string,
): Promise<PdfAcquisition> {
  return coreRequest(connection, `/pdf-acquisitions/${encodeURIComponent(acquisitionId)}/retry`, {
    method: "POST",
  });
}

export function cancelPdfAcquisition(
  connection: CoreConnection,
  acquisitionId: string,
): Promise<PdfAcquisition> {
  return coreRequest(connection, `/pdf-acquisitions/${encodeURIComponent(acquisitionId)}/cancel`, {
    method: "POST",
  });
}

export function listPdfImports(
  connection: CoreConnection,
  sourceId: string,
): Promise<PdfImport[]> {
  return coreRequest(connection, `/sources/${encodeURIComponent(sourceId)}/imports`);
}

export function trashPdfImport(
  connection: CoreConnection,
  sourceId: string,
  importId: string,
): Promise<PdfImportTrash> {
  return coreRequest(
    connection,
    `/sources/${encodeURIComponent(sourceId)}/imports/${encodeURIComponent(importId)}`,
    { method: "DELETE" },
  );
}

export function retryPdfProcessing(
  connection: CoreConnection,
  sourceId: string,
  importId: string,
): Promise<ProcessingAction> {
  return coreRequest(
    connection,
    `/sources/${encodeURIComponent(sourceId)}/imports/${encodeURIComponent(importId)}/retry`,
    { method: "POST" },
  );
}

export function cancelPdfProcessing(
  connection: CoreConnection,
  sourceId: string,
  importId: string,
): Promise<ProcessingAction> {
  return coreRequest(
    connection,
    `/sources/${encodeURIComponent(sourceId)}/imports/${encodeURIComponent(importId)}/cancel`,
    { method: "POST" },
  );
}

export function trashSource(
  connection: CoreConnection,
  sourceId: string,
): Promise<PdfImportTrash> {
  return coreRequest(connection, `/sources/${encodeURIComponent(sourceId)}`, {
    method: "DELETE",
  });
}

export function storageReport(connection: CoreConnection): Promise<StorageReport> {
  return coreRequest(connection, "/storage/originals");
}

export function pdfImportUrl(connection: CoreConnection, sourceId: string, importId: string): string {
  return `${connection.base_url}/sources/${encodeURIComponent(sourceId)}/imports/${encodeURIComponent(importId)}/pdf`;
}

function errorDetail(value: unknown): CoreApiErrorDetail {
  if (typeof value !== "object" || value === null || !("detail" in value)) return {};
  const detail = (value as { detail?: unknown }).detail;
  if (typeof detail === "string") return { message: detail };
  if (typeof detail !== "object" || detail === null) return {};
  const candidate = detail as CoreApiErrorDetail;
  return {
    code: typeof candidate.code === "string" ? candidate.code : undefined,
    message: typeof candidate.message === "string" ? candidate.message : undefined,
  };
}

export function coreHealth(connection: CoreConnection): Promise<CoreHealth> {
  return coreRequest<CoreHealth>(connection, "/health");
}

export async function coreHealthWithRetry(
  connection: CoreConnection,
  attempts = 5,
  delayMs = 500,
): Promise<CoreHealth> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await coreHealth(connection);
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
  }
  throw lastError;
}

export function getExtraction(
  connection: CoreConnection,
  sourceId: string,
): Promise<ExtractionReport> {
  return coreRequest(connection, `/sources/${encodeURIComponent(sourceId)}/extraction`);
}

export function decideExtractionReview(
  connection: CoreConnection,
  extractionId: string,
  payload: { decision: "confirmed" | "declined"; generation: number },
): Promise<ExtractionReview> {
  return coreRequest(connection, `/extractions/${encodeURIComponent(extractionId)}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function reopenExtractionReview(
  connection: CoreConnection,
  extractionId: string,
): Promise<ExtractionReview> {
  return coreRequest(connection, `/extractions/${encodeURIComponent(extractionId)}/review/reopen`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
}
