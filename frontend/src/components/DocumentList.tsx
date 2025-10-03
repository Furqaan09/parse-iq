import { useEffect, useState } from "react"
import { listDocuments, type DocumentItem, getDocument } from "../lib/api"

export default function DocumentList() {
    const [items, setItems] = useState<DocumentItem[]>([])
    const [page, setPage] = useState(1)
    const [total, setTotal] = useState(0)
    const [pageSize, setPageSize] = useState(20)
    const [hasNext, setHasNext] = useState(false)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [query, setQuery] = useState("")

    async function load(p = page, q = query) {
        setLoading(true)
        setError(null)
        try {
            const data = await listDocuments({ page: p, page_size: pageSize, q })
            setItems(data.items)
            setTotal(data.total)
            setHasNext(data.has_next)
            setPage(p)
        } catch (e: any) {
            setError(e?.message ?? "Failed to load documents")
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => { load(1, "") }, []) // initial load

    async function handleOpen(id: number) {
        try {
            const doc = await getDocument(id)
            if (!doc.file_url) {
                alert("No file URL available for this document.")
                return
            }

            // If the API returns an absolute URL (e.g., S3 later), use it as-is.
            const isAbsolute = /^https?:\/\//i.test(doc.file_url)
            const apiOrigin = import.meta.env.VITE_API_ORIGIN
            const url = isAbsolute ? doc.file_url : `${apiOrigin}${doc.file_url}`

            // Open in a new tab (served by FastAPI static mount)
            window.open(url, "_blank", "noopener,noreferrer")
        } catch (e: any) {
            alert(e?.message ?? "Failed to open document")
        }
    }

    return (
        <section className="mt-6 rounded-2xl border bg-white p-6 shadow-sm">
            <div className="mb-4 flex items-center gap-2">
                <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search title..."
                    className="w-full rounded-lg border px-3 py-2 text-sm"
                />
                <button
                    onClick={() => load(1, query)}
                    className="rounded-lg text-white border px-3 py-2 text-sm hover:bg-gray-50"
                >
                    Search
                </button>
                <button
                    onClick={() => { setQuery(""); load(1, "") }}
                    className="rounded-lg text-white border px-3 py-2 text-sm hover:bg-gray-50"
                >
                    Reset
                </button>
            </div>

            {loading && <div className="text-sm text-gray-600">Loading…</div>}
            {error && <div className="rounded-md border border-red-200 bg-red-50 p-2 text-sm text-red-700">{error}</div>}

            {!loading && !error && items.length === 0 && (
                <div className="text-sm text-gray-600">No documents yet. Upload something above.</div>
            )}

            <ul className="grid gap-3 sm:grid-cols-2">
                {items.map((d) => (
                    <li key={d.id} className="rounded-xl border p-4">
                        <div className="flex items-start justify-between">
                            <div>
                                <div className="font-semibold">{d.title}</div>
                                <div className="text-xs text-gray-600">
                                    {d.media_type.toUpperCase()} {d.pages != null ? `• ${d.pages} page(s)` : ""}
                                </div>
                            </div>
                            <button
                                onClick={() => handleOpen(d.id)}
                                className="rounded-lg text-white border px-3 py-1.5 text-sm hover:bg-gray-50"
                            >
                                Open
                            </button>
                        </div>
                        <div className="mt-2 truncate text-xs text-gray-500">{d.storage_path}</div>
                    </li>
                ))}
            </ul>

            <div className="mt-4 flex items-center justify-between">
                <div className="text-xs text-gray-600">
                    Showing {items.length} of {total}
                </div>
                <div className="flex items-center gap-2">
                    <button
                        disabled={page <= 1 || loading}
                        onClick={() => load(page - 1, query)}
                        className="rounded-lg text-white border px-3 py-1.5 text-sm disabled:opacity-50"
                    >
                        Prev
                    </button>
                    <span className="text-sm">Page {page}</span>
                    <button
                        disabled={!hasNext || loading}
                        onClick={() => load(page + 1, query)}
                        className="rounded-lg text-white border px-3 py-1.5 text-sm disabled:opacity-50"
                    >
                        Next
                    </button>
                </div>
            </div>
        </section>
    )
}