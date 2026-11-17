import { expect, it } from "vitest";
import { pageLayout, TILE_SIDE, visibleTiles } from "./pdfTiles";

it("keeps the original near-20-MP resolution rather than lowering it to fit tiles", () => {
  const layout = pageLayout(1700 * 2.7, 1700 * 2.7, 1.25, 20_000_000);
  expect(layout.width * layout.height).toBeGreaterThanOrEqual(19_000_000);
  expect(layout.width * layout.height).toBeLessThanOrEqual(20_000_000);
  expect(layout.cssWidth).toBe(4590);
  const tiles = visibleTiles(layout, { left: 0, top: 0, width: 1000, height: 600 });
  expect(tiles.length).toBeGreaterThan(1);
  expect(tiles.every((tile) => tile.width <= TILE_SIDE && tile.height <= TILE_SIDE)).toBe(true);
  expect(tiles.reduce((total, tile) => total + tile.width * tile.height, 0)).toBeLessThan(2_000_000);
});

it("covers a scrolled region across tile boundaries in physical rather than CSS pixels", () => {
  const layout = pageLayout(1000, 1000, 1.25, 20_000_000);
  const tiles = visibleTiles(layout, { left: 400, top: 400, width: 20, height: 20 });
  expect(tiles.map((tile) => tile.id)).toEqual(["0:0", "512:0", "0:512", "512:512"]);
  expect(tiles[3]).toMatchObject({ x: 512, y: 512, width: 512, height: 512 });
});

it("clips edge tiles exactly and never allocates overlapping full-page backing", () => {
  const layout = pageLayout(1100, 700, 1, 20_000_000);
  const tiles = visibleTiles(layout, { left: 0, top: 0, width: 1100, height: 700 });
  expect(tiles).toHaveLength(6);
  expect(tiles[5]).toMatchObject({ x: 1024, y: 512, width: 76, height: 188 });
  expect(tiles.reduce((total, tile) => total + tile.width * tile.height, 0)).toBe(layout.width * layout.height);
  expect(visibleTiles(layout, { left: 1200, top: 0, width: 20, height: 20 })).toEqual([]);
});

it("rejects impossible geometry instead of increasing the page limit", () => {
  expect(() => pageLayout(Infinity, 100, 1, 20_000_000)).toThrow("dimensions");
  expect(() => pageLayout(100, 100, 0, 20_000_000)).toThrow("dimensions");
  expect(() => visibleTiles(pageLayout(100, 100, 1, 20_000_000), { left: NaN, top: 0, width: 20, height: 20 })).toThrow("invalid");
});
