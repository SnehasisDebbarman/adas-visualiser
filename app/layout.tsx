import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ADAS Visualiser",
  description: "Browser-based ADAS perception visualiser for live camera and uploaded driving footage."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
