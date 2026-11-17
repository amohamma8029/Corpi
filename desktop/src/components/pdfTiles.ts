// Physical tile geometry with the same full-page resolution policy.
export const TILE_SIDE = 512;

export interface PageLayout {
  width: number;
  height: number;
  cssWidth: number;
  cssHeight: number;
  outputScale: number;
}

export interface Tile {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface VisibleBox { left: number; top: number; width: number; height: number }

export function pageLayout(cssWidth: number, cssHeight: number, deviceScale: number, pixelLimit: number): PageLayout {
  if (![cssWidth, cssHeight, deviceScale, pixelLimit].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error("The PDF page dimensions are not supported.");
  }
  const outputScale = Math.min(deviceScale, Math.sqrt(pixelLimit / (cssWidth * cssHeight)) * 0.999);
  const width = Math.max(1, Math.ceil(cssWidth * outputScale));
  const height = Math.max(1, Math.ceil(cssHeight * outputScale));
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width * height > pixelLimit) {
    throw new Error("The PDF page exceeds its rendered-pixel limit.");
  }
  return { width, height, cssWidth, cssHeight, outputScale };
}

export function visibleTiles(layout: PageLayout, box: VisibleBox): Tile[] {
  if (![box.left, box.top, box.width, box.height].every(Number.isFinite) || box.width < 0 || box.height < 0) {
    throw new Error("The visible PDF region is invalid.");
  }
  const scaleX = layout.width / layout.cssWidth;
  const scaleY = layout.height / layout.cssHeight;
  const left = Math.max(0, box.left * scaleX);
  const top = Math.max(0, box.top * scaleY);
  const right = Math.min(layout.width, (box.left + box.width) * scaleX);
  const bottom = Math.min(layout.height, (box.top + box.height) * scaleY);
  if (right <= left || bottom <= top) return [];
  const tiles: Tile[] = [];
  for (let y = Math.floor(top / TILE_SIDE) * TILE_SIDE; y < bottom; y += TILE_SIDE) {
    for (let x = Math.floor(left / TILE_SIDE) * TILE_SIDE; x < right; x += TILE_SIDE) {
      tiles.push({ id: `${x}:${y}`, x, y,
        width: Math.min(TILE_SIDE, layout.width - x), height: Math.min(TILE_SIDE, layout.height - y) });
    }
  }
  return tiles;
}
