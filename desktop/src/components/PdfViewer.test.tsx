import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { PdfViewer } from "./PdfViewer";

vi.mock("../api/client", () => ({ coreBinary: async () => new Uint8Array([37, 80, 68, 70]) }));
vi.mock("pdfjs-dist/legacy/build/pdf.worker.mjs?worker", () => ({
  default: class { terminate() {} },
}));
vi.mock("pdfjs-dist/legacy/build/pdf.mjs", () => ({
  PDFWorker: { create: () => ({ destroy() {} }) },
  getDocument: () => ({
    destroy: async () => {},
    promise: Promise.resolve({
      numPages: 2,
      getPage: async (number: number) => ({
        getViewport: ({ scale }: { scale: number }) => ({ width: 100 * scale, height: 100 * scale }),
        render: () => ({
          cancel() {},
          promise: number === 1 ? Promise.reject(new Error("Page one cannot render")) : Promise.resolve(),
        }),
      }),
    }),
  }),
}));

const originalShowModal = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, "showModal");
const originalClose = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, "close");
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  if (originalShowModal) Object.defineProperty(HTMLDialogElement.prototype, "showModal", originalShowModal);
  else Reflect.deleteProperty(HTMLDialogElement.prototype, "showModal");
  if (originalClose) Object.defineProperty(HTMLDialogElement.prototype, "close", originalClose);
  else Reflect.deleteProperty(HTMLDialogElement.prototype, "close");
});

it("keeps the canvas usable after a page error and clears the error when another page renders", async () => {
  // JSDOM has no modal API; actual focus/inert behavior is checked in native E2E.
  Object.defineProperties(HTMLDialogElement.prototype, {
    showModal: { configurable: true, value: function (this: HTMLDialogElement) { this.open = true; } },
    close: { configurable: true, value: function (this: HTMLDialogElement) { this.open = false; } },
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({} as CanvasRenderingContext2D);
  render(<PdfViewer connection={{ base_url: "http://127.0.0.1:9", token: "test-token" }}
    sourceId="source" importId="import" filename="two-pages.pdf" renderedPagePixelLimit={20_000_000}
    onClose={() => {}} />);
  expect((await screen.findByRole("alert")).textContent).toContain("Page one cannot render");
  const canvas = document.querySelector("canvas")!;
  expect(canvas.parentElement!.hidden).toBe(false);
  const close = screen.getByRole("button", { name: "Close PDF" });
  const pageInput = screen.getByRole("spinbutton", { name: "Page number" });
  close.focus();
  fireEvent.keyDown(close, { key: "Tab" });
  expect(document.activeElement).toBe(pageInput); // skip disabled Previous page
  fireEvent.keyDown(pageInput, { key: "Tab", shiftKey: true });
  expect(document.activeElement).toBe(close);
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await waitFor(() => expect(canvas.parentElement!.getAttribute("aria-busy")).toBe("false"));
  expect(screen.queryByRole("alert")).toBeNull();
  expect(document.querySelector("canvas")).toBe(canvas);
});
