export interface PdfFixture {
  filename: string;
  kind: "text" | "search-text" | "raster" | "invalid";
}

// Self-contained so the same function can construct the File inside the webview.
// Passing only a descriptor avoids multi-megabyte WebDriver request bodies.
export function pdfFixtureBytes(fixture: PdfFixture): Uint8Array {
  if (fixture.kind === "invalid") {
    return new TextEncoder().encode("%PDF-1.4\nCorpi invalid viewer fixture\n%%EOF\n");
  }
  let objects: string[];
  if (fixture.kind === "text" || fixture.kind === "search-text") {
    const searchable = "BT /F1 8 Tf 15 150 Td 12 TL (The experiment measured rainfall) Tj T* (and soil moisture across study sites.) Tj T* (Researchers compared daily readings) Tj T* (to understand the seasonal changes.) Tj ET";
    const first = "q 0.2 0.5 0.8 rg 0 0 200 200 re f Q\n" + (fixture.kind === "search-text" ? searchable : "BT /F1 18 Tf 20 100 Td (Corpi page one) Tj ET");
    const second = "q 0.8 0.2 0.3 rg 0 0 200 200 re f Q\n" + (fixture.kind === "search-text" ? searchable : "BT /F1 18 Tf 20 100 Td (Corpi page two) Tj ET");
    objects = [
      "<< /Type /Catalog /Pages 2 0 R >>",
      "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >>",
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>",
      `<< /Length ${first.length} >>\nstream\n${first}\nendstream`,
      `<< /Length ${second.length} >>\nstream\n${second}\nendstream`,
      "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ];
  } else {
    const stream = "q 1700 0 0 1700 0 0 cm /Image1 Do Q";
    const image = `${"cc8855".repeat(512 * 512)}>`;
    objects = [
      "<< /Type /Catalog /Pages 2 0 R >>",
      "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1700 1700] /Contents 4 0 R /Resources << /XObject << /Image1 5 0 R >> >> >>",
      `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`,
      `<< /Type /XObject /Subtype /Image /Width 512 /Height 512 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length ${image.length} /Filter /ASCIIHexDecode >>\nstream\n${image}\nendstream`,
    ];
  }
  let body = "%PDF-1.4\n";
  const offsets = [0];
  objects.forEach((object, index) => {
    offsets.push(body.length);
    body += `${index + 1} 0 obj\n${object}\nendobj\n`;
  });
  if (fixture.kind === "raster") {
    // Valid PDF comment padding before xref exercises a full-file near-25-MB transfer.
    body += `%${"p".repeat(24_999_000 - body.length)}\n`;
  }
  const xref = body.length;
  body += `xref\n0 ${offsets.length}\n0000000000 65535 f \n`;
  for (const offset of offsets.slice(1)) body += `${String(offset).padStart(10, "0")} 00000 n \n`;
  body += `trailer\n<< /Size ${offsets.length} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return new TextEncoder().encode(body);
}
