export interface SurahSummary {
  id: number;
  slug: string;
  name_ar: string;
  name_en: string;
  ayah_count: number;
  enabled: boolean;
}

export interface SurahWord {
  word_index: number;
  global_word_index: number;
  text: string;
}

export interface SurahAyah {
  ayah_index: number;
  text_ar: string;
  words: SurahWord[];
}

export interface SurahDetailResponse {
  surah: {
    id: number;
    slug: string;
    name_ar: string;
    name_en: string;
    ayah_count: number;
  };
  ayahs: SurahAyah[];
  total_words: number;
}

const API_BASE = "http://localhost:8000";

export async function fetchSurahs(): Promise<SurahSummary[]> {
  const res = await fetch(`${API_BASE}/api/recitation/surahs`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch surah list");
  const data = await res.json();
  return data.surahs ?? [];
}

export async function fetchSurahDetail(slug: string): Promise<SurahDetailResponse> {
  const res = await fetch(`${API_BASE}/api/recitation/surahs/${slug}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch surah content");
  return res.json();
}
