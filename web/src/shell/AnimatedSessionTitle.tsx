import { useEffect, useRef, useState } from "react";

const MAX_REVEAL_MS = 1_200;
const MIN_CHARACTER_MS = 32;
const MAX_CHARACTER_MS = 48;

export function AnimatedSessionTitle({
  title,
  fallback,
}: {
  title: string | null | undefined;
  fallback: string;
}) {
  const previousTitle = useRef(title);
  const [displayedTitle, setDisplayedTitle] = useState(title ?? fallback);
  const [animating, setAnimating] = useState(false);

  useEffect(() => {
    const previous = previousTitle.current;
    previousTitle.current = title;

    if (title == null) {
      setDisplayedTitle(fallback);
      setAnimating(false);
      return;
    }

    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
    if (previous != null || reduceMotion) {
      setDisplayedTitle(title);
      setAnimating(false);
      return;
    }

    let characterCount = 0;
    const characterMs = Math.max(
      MIN_CHARACTER_MS,
      Math.min(MAX_CHARACTER_MS, Math.floor(MAX_REVEAL_MS / title.length)),
    );
    setDisplayedTitle("");
    setAnimating(true);
    const timer = window.setInterval(() => {
      characterCount += 1;
      setDisplayedTitle(title.slice(0, characterCount));
      if (characterCount >= title.length) {
        window.clearInterval(timer);
        setAnimating(false);
      }
    }, characterMs);
    return () => window.clearInterval(timer);
  }, [fallback, title]);

  return (
    <span aria-label={title ?? fallback}>
      <span aria-hidden="true">{displayedTitle}</span>
      {animating && (
        <span aria-hidden="true" className="ml-px animate-pulse text-muted-foreground">
          ▍
        </span>
      )}
    </span>
  );
}
