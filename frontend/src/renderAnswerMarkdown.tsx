import { Fragment, type ReactNode } from "react";

function boldSegments(text: string): ReactNode {
  const re = /\*\*(.+?)\*\*/g;
  const parts: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      parts.push(text.slice(last, m.index));
    }
    parts.push(<strong key={`b${k++}`}>{m[1]}</strong>);
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    parts.push(text.slice(last));
  }
  return parts.length ? parts : text;
}

function paragraphBlock(text: string, key: number): ReactNode {
  const lines = text.split("\n");
  return (
    <p key={key} className="query-bar__md-p">
      {lines.map((line, i) => (
        <Fragment key={i}>
          {i > 0 ? <br /> : null}
          {boldSegments(line)}
        </Fragment>
      ))}
    </p>
  );
}

function listBlock(block: string, key: number): ReactNode {
  const lines = block.split("\n").filter((l) => l.trim());
  return (
    <ul key={key} className="query-bar__md-ul">
      {lines.map((line, i) => {
        const item = line.trim().replace(/^-\s+/, "");
        return (
          <li key={i} className="query-bar__md-li">
            {boldSegments(item)}
          </li>
        );
      })}
    </ul>
  );
}

function isListBlock(block: string): boolean {
  const lines = block.split("\n").filter((l) => l.trim());
  return lines.length > 0 && lines.every((l) => /^\s*-\s+/.test(l));
}

export function renderAnswerMarkdown(text: string): ReactNode {
  const blocks = text.trim().split(/\n{2,}/);
  return blocks.map((raw, bi) => {
    const b = raw.trim();
    if (!b) return null;
    if (isListBlock(b)) {
      return listBlock(b, bi);
    }
    return paragraphBlock(b, bi);
  });
}
