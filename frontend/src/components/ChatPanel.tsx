import { useState } from "react"
import { askChat, type ChatCitation } from "../lib/api"

type ChatMessage =
    | {
        role: "user"
        content: string
    }
    | {
        role: "assistant"
        content: string
        citations: ChatCitation[]
    }

const API_ORIGIN = import.meta.env.VITE_API_ORIGIN || "http://localhost:8000"

export default function ChatPanel() {
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [input, setInput] = useState("")
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault()

        const trimmed = input.trim()
        if (!trimmed || isLoading) return

        setError(null)

        const userMessage: ChatMessage = {
            role: "user",
            content: trimmed,
        }

        setMessages((prev) => [...prev, userMessage])
        setInput("")
        setIsLoading(true)

        try {
            const result = await askChat({
                message: trimmed,
                top_k: 6,
            })

            const assistantMessage: ChatMessage = {
                role: "assistant",
                content: result.answer,
                citations: result.citations,
            }

            setMessages((prev) => [...prev, assistantMessage])
        } catch (err: any) {
            setError(err?.message ?? "Something went wrong while asking the assistant.")
        } finally {
            setIsLoading(false)
        }
    }

    function openCitation(fileUrl: string | null) {
        if (!fileUrl) return
        const isAbsolute = /^https?:\/\//i.test(fileUrl)
        const fullUrl = isAbsolute ? fileUrl : `${API_ORIGIN}${fileUrl}`
        window.open(fullUrl, "_blank", "noopener,noreferrer")
    }

    return (
        <section className="mt-6 rounded-2xl border bg-white p-6 shadow-sm">
            <div className="mb-4">
                <h2 className="text-xl font-semibold">Ask ParseIQ</h2>
                <p className="text-sm text-gray-600">
                    Ask questions about your uploaded documents.
                </p>
            </div>

            <div className="mb-4 max-h-[500px] space-y-4 overflow-y-auto rounded-xl border bg-gray-50 p-4">
                {messages.length === 0 && (
                    <div className="text-sm text-gray-500">
                        Try asking something like:
                        <div className="mt-2">“What is the late penalty?”</div>
                        <div>“List the important deadlines.”</div>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div
                        key={idx}
                        className={`rounded-xl p-4 ${msg.role === "user"
                                ? "ml-auto max-w-[85%] border bg-blue-50"
                                : "mr-auto max-w-[90%] border bg-white"
                            }`}
                    >
                        <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">
                            {msg.role === "user" ? "You" : "ParseIQ"}
                        </div>

                        <div className="whitespace-pre-wrap text-sm text-gray-900">
                            {msg.content}
                        </div>

                        {msg.role === "assistant" && msg.citations.length > 0 && (
                            <div className="mt-4 space-y-2 border-t pt-3">
                                <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                                    Citations
                                </div>

                                {msg.citations.map((citation) => (
                                    <div
                                        key={`${idx}-${citation.n}`}
                                        className="rounded-lg border bg-gray-50 p-3"
                                    >
                                        <div className="flex items-start justify-between gap-3">
                                            <div>
                                                <div className="text-sm font-medium">
                                                    [{citation.n}] {citation.title}
                                                </div>
                                                <div className="text-xs text-gray-500">
                                                    Page {citation.page ?? "—"}
                                                </div>
                                            </div>

                                            <button
                                                type="button"
                                                onClick={() => openCitation(citation.file_url)}
                                                disabled={!citation.file_url}
                                                className="rounded-lg border px-3 py-1.5 text-xs hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
                                            >
                                                Open source
                                            </button>
                                        </div>

                                        <div className="mt-2 text-xs text-gray-700">
                                            {citation.snippet}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                ))}

                {isLoading && (
                    <div className="mr-auto max-w-[90%] rounded-xl border bg-white p-4">
                        <div className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-500">
                            ParseIQ
                        </div>
                        <div className="text-sm text-gray-600">Thinking…</div>
                    </div>
                )}
            </div>

            {error && (
                <div className="mb-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                    {error}
                </div>
            )}

            <form onSubmit={handleSubmit} className="flex gap-3">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a question about your documents..."
                    className="flex-1 rounded-xl border px-4 py-3 text-sm outline-none focus:border-gray-400"
                    disabled={isLoading}
                />
                <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    className="rounded-xl border px-4 py-3 text-sm font-medium hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                    Send
                </button>
            </form>
        </section>
    )
}