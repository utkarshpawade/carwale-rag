export type Car = {
  id: string;
  name: string;
  url: string;
  image?: string | null;
  price: string;
  body?: string | null;
  rating?: number | null;
  status?: string;
  ncap?: number | null;
};

export type Source = {
  n: number;
  car: string;
  section: string;
  url?: string | null;
  model_id?: string | null;
  kind?: string;
  score?: number;
  via?: string[];
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Trace = Record<string, any>;

export type ChatMessage = {
  key: string;
  id?: number;
  role: "user" | "assistant";
  content: string;
  status?: string;
  cars?: Car[];
  sources?: Source[];
  followups?: string[];
  trace?: Trace;
  error?: string;
  streaming?: boolean;
  cached?: boolean;
  feedback?: number;
};

export type Conversation = { id: string; title: string; updated_at: string; n_messages: number };
