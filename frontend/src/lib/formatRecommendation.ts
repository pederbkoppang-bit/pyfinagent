export function formatRecommendation(rec: string | null | undefined): string {
  if (!rec || !rec.trim()) return "—";
  return rec.toUpperCase().replace(/_/g, " ");
}
