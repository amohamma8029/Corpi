/** Model identities, separate from the descriptions of what each option is for. */
export function modelName(profileId: string): string {
  const names: Record<string, string> = {
    "bekko-a8m-quint8-avx2": "Bekko A8M (quint8 AVX2)",
    "minilm-en": "MiniLM-L6-v2",
    "multilingual-e5": "Multilingual E5-small",
  };
  return names[profileId] ?? profileId;
}
