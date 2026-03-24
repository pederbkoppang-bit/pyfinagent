"use client";

import { useState, useRef, useEffect } from "react";
import { BentoCard } from "./BentoCard";
import { IconSearch } from "@/lib/icons";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function ResearchInvestigator({ ticker }: { ticker: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/investigate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, question: q }),
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status}: ${body}`);
      }

      const data = await res.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${e instanceof Error ? e.message : "Unknown error"}. The research investigator endpoint may not be available yet.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <BentoCard className="flex h-full flex-col">
      <div className="mb-3 flex items-center gap-2">
        <IconSearch size={20} weight="duotone" className="text-sky-400" />
        <h3 className="text-lg font-semibold text-slate-200">Research Investigator</h3>
      </div>

      {/* Messages area */}
      <div
        ref={scrollRef}
        className="flex-1 space-y-3 overflow-y-auto scrollbar-thin rounded-lg bg-slate-900/50 p-3"
        style={{ maxHeight: "calc(100% - 6rem)", minHeight: 200 }}
      >
        {messages.length === 0 && (
          <p className="text-sm text-slate-500">
            Ask a follow-up question about the {ticker || "company"} analysis.
          </p>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`rounded-lg p-3 text-sm ${
              msg.role === "user"
                ? "ml-8 bg-sky-500/10 text-sky-200"
                : "mr-8 bg-slate-800 text-slate-300"
            }`}
          >
            <p className="mb-1 text-[10px] font-medium uppercase text-slate-500">
              {msg.role === "user" ? "You" : "AI Investigator"}
            </p>
            <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
          </div>
        ))}
        {loading && (
          <div className="mr-8 flex items-center gap-2 rounded-lg bg-slate-800 p-3">
            <div className="gemini-spinner">
              <div className="gemini-bar" />
              <div className="gemini-bar" />
              <div className="gemini-bar" />
              <div className="gemini-bar" />
            </div>
            <span className="text-xs text-slate-500">Investigating...</span>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="mt-3 flex items-center gap-2 rounded-lg border border-slate-700 p-1.5">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder={`e.g., "Explain the P/E ratio for ${ticker || 'this stock'}..."`}
          className="flex-1 bg-transparent px-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="rounded-md bg-sky-500 p-2 text-white transition-colors hover:bg-sky-400 disabled:opacity-50"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
    </BentoCard>
  );
}
