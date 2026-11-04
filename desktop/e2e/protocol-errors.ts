/** Recognize the two result shapes emitted by the pinned WebDriver client. */
export function isStaleElementResult(result: unknown): boolean {
  if (!result || typeof result !== "object") return false;
  return [Reflect.get(result, "error"), Reflect.get(result, "value")].some((value) =>
    value && typeof value === "object"
      && (Reflect.get(value, "name") === "stale element reference"
        || Reflect.get(value, "error") === "stale element reference"));
}
