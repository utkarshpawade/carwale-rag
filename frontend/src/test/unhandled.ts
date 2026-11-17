/** Collects unhandled promise rejections while a test runs (jsdom does not forward Node's to `window`). */
type Proc = { on(e: string, f: (r: unknown) => void): void; off(e: string, f: (r: unknown) => void): void };

export function captureUnhandledRejections() {
  const proc = (globalThis as unknown as { process: Proc }).process;
  const reasons: unknown[] = [];
  const onNode = (r: unknown) => { reasons.push(r); };
  const onWindow = (e: PromiseRejectionEvent) => { reasons.push(e.reason); };
  proc.on("unhandledRejection", onNode);
  window.addEventListener("unhandledrejection", onWindow);
  return {
    reasons,
    stop: () => {
      proc.off("unhandledRejection", onNode);
      window.removeEventListener("unhandledrejection", onWindow);
    },
  };
}

export const settle = (ms = 30) => new Promise((r) => setTimeout(r, ms));
