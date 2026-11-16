type P = { className?: string };
const base = (d: string) => ({ className = "w-4 h-4" }: P) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
    <path d={d} />
  </svg>
);

export const PlusIcon = base("M12 5v14M5 12h14");
export const TrashIcon = base("M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6");
export const MenuIcon = base("M3 6h18M3 12h18M3 18h18");
export const SendIcon = base("M22 2 11 13M22 2l-7 20-4-9-9-4 20-7z");
export const StopIcon = base("M6 6h12v12H6z");
export const CopyIcon = base("M9 9h11v11H9zM5 15H4V4h11v1");
export const UpIcon = base("M7 10v11M2 10h5v11H2zM7 10l4-8a3 3 0 0 1 3 3v4h5.5a2 2 0 0 1 2 2.3l-1.4 7A2 2 0 0 1 18 21H7");
export const DownIcon = base("M17 14V3M22 14h-5V3h5zM17 14l-4 8a3 3 0 0 1-3-3v-4H4.5a2 2 0 0 1-2-2.3l1.4-7A2 2 0 0 1 6 3h11");
export const XIcon = base("M18 6 6 18M6 6l12 12");
export const FlowIcon = base("M4 6h6v4H4zM14 14h6v4h-6zM7 10v4a2 2 0 0 0 2 2h5");
export const ExternalIcon = base("M14 3h7v7M10 14 21 3M21 14v7H3V3h7");
export const SparkIcon = base("M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z");
