import { expect, it } from "vitest";

import { isStaleElementResult } from "../e2e/protocol-errors";

it("recognizes rejected and returned stale-element protocol results", () => {
  const error = new Error("Element is no longer attached to the DOM");
  error.name = "stale element reference";
  expect(isStaleElementResult({ error })).toBe(true);
  expect(isStaleElementResult({ value: { error: "stale element reference" } })).toBe(true);
});

it.each([null, {}, { value: null }, { value: "stale element reference" },
  { error: new Error("Application failure") }, { value: { error: "no such element" } }])(
  "does not misidentify unrelated results: %j", (result) => {
    expect(isStaleElementResult(result)).toBe(false);
  },
);
