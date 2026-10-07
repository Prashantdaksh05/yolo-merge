import "./globals.css";
export const metadata = { title: "YOLO Merge", description: "Run two YOLO models on one image set and merge the labels." };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
