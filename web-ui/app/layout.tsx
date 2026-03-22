import type { Metadata } from "next";
import { Amiri, Space_Mono, Syne } from "next/font/google";
import "./globals.css";

const amiri = Amiri({
  weight: ["400", "700"],
  subsets: ["arabic", "latin"],
  variable: "--font-amiri",
});

const spaceMono = Space_Mono({
  weight: ["400", "700"],
  subsets: ["latin"],
  variable: "--font-mono",
});

const syne = Syne({
  weight: ["400", "700", "800"],
  subsets: ["latin"],
  variable: "--font-syne",
});

export const metadata: Metadata = {
  title: "Quran Live Transcriber",
  description: "Real-time Arabic Quran transcription powered by fine-tuned Whisper",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${amiri.variable} ${spaceMono.variable} ${syne.variable}`}>
      <body className="min-h-screen bg-[#0a0a0f] text-[#e8e4db]">
        {children}
      </body>
    </html>
  );
}
