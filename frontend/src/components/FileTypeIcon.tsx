/** File extension including dot, e.g. `.pdf` */
export function fileExt(filename: string): string {
  const i = filename.lastIndexOf(".");
  return i >= 0 ? filename.slice(i).toLowerCase() : "";
}

type Props = {
  ext: string;
  size?: "md" | "lg";
  /** Generic “any document” glyph for empty drop state */
  placeholder?: boolean;
};

/** Document glyph + format badge */
export function FileTypeIcon({ ext, size = "md", placeholder }: Props) {
  const kind = placeholder
    ? "any"
    : ext === ".pdf"
      ? "pdf"
      : ext === ".docx"
        ? "docx"
        : ext === ".md"
          ? "md"
          : ext === ".txt"
            ? "txt"
            : "txt";
  const label = placeholder ? "FILE" : kind.toUpperCase();
  const dim = size === "lg" ? { w: 36, h: 44, fs: "0.58rem" } : { w: 28, h: 34, fs: "0.5rem" };

  return (
    <span className={`file-type-icon file-type-icon--${kind} file-type-icon--${size}`} title={ext || "document"} aria-hidden>
      <svg
        className="file-type-icon__sheet"
        width={dim.w}
        height={dim.h}
        viewBox="0 0 32 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M4 3a2 2 0 0 1 2-2h14l10 10v26a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V3Z"
          fill="currentColor"
          fillOpacity="0.12"
        />
        <path
          d="M20 1v8a2 2 0 0 0 2 2h8"
          stroke="currentColor"
          strokeOpacity="0.4"
          strokeWidth="1.2"
          strokeLinejoin="round"
        />
        <path d="M8 18h16M8 24h12M8 30h14" stroke="currentColor" strokeOpacity="0.22" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
      <span className="file-type-icon__badge" style={{ fontSize: dim.fs }}>
        {label}
      </span>
    </span>
  );
}
