# Temporary application icon

`icon-source.svg` is a temporary placeholder, not a settled brand identity. It was
drawn for Corpi from a blue rectangle and a white circular arc. It uses no font,
external image, or third-party artwork and is covered by the root MIT license.

The SVG is the editable source. `icon-source.png` is its 1024-pixel raster preview;
the files under `../src-tauri/icons/` are generated platform variants. The earlier
icon's origin remains unconfirmed in Git history; this replacement establishes
the source of the current assets only.

Use the repository's locked Tauri CLI. From the repository root, regenerate the
platform variants with:

```powershell
pnpm --dir desktop exec tauri icon assets/icon-source.svg
```

For the raster preview, render into a new temporary directory:

```powershell
pnpm --dir desktop exec tauri icon assets/icon-source.svg --png 1024 --output <temporary-directory>
```

Copy its `1024x1024.png` to `desktop/assets/icon-source.png`. Check the generated
file list before committing; a future tool update may introduce new variants.
The current CLI may reorder ICNS container blocks between runs; its decoded images
and block payloads are repeatable even when the whole-file hash differs.
