import { useRef, useState } from "react"
import { uploadDocument, type UploadResponse } from "../lib/api"

export default function UploadBox() {
    const inputRef = useRef<HTMLInputElement | null>(null)
    const [fileName, setFileName] = useState<string>("")
    const [isUploading, setIsUploading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [result, setResult] = useState<UploadResponse | null>(null)

    async function handleUpload() {
        // Reset previous states
        setError(null)
        setResult(null)

        // Get the selected file
        const file = inputRef.current?.files?.[0]
        if (!file) {
            setError("Please choose a file first.")
            return
        }

        // Upload the file
        try {
            setIsUploading(true)
            const data = await uploadDocument(file)
            setResult(data)
        } catch (e: any) {
            setError(e?.message ?? "Upload failed")
        } finally {
            setIsUploading(false)
        }
    }

    return (
        <div className="space-y-3 rounded-2xl border bg-white p-4 shadow-sm">
            <div className="text-lg font-semibold">Upload a document</div>

            <input
                ref={inputRef}
                type="file"
                onChange={(e) => setFileName(e.target.files?.[0]?.name ?? "")}
                className="block w-full"
                aria-label="Choose file"
            />

            <div className="flex items-center gap-3">
                <button
                    onClick={handleUpload}
                    disabled={isUploading}
                    className="rounded-lg text-white border px-3 py-1.5 text-sm disabled:cursor-not-allowed disabled:opacity-50"
                >
                    {isUploading ? "Uploading..." : "Upload"}
                </button>
                {fileName && <span className="text-sm text-gray-600">{fileName}</span>}
            </div>

            {error && (
                <div className="rounded-md border border-red-200 bg-red-50 p-2 text-sm text-red-700">
                    {error}
                </div>
            )}

            {result && (
                <div className="rounded-md border bg-gray-50 p-3 text-sm">
                    <div><span className="font-medium">ID:</span> {result.id}</div>
                    <div><span className="font-medium">Title:</span> {result.title}</div>
                    <div><span className="font-medium">Media:</span> {result.media_type}</div>
                    <div><span className="font-medium">Pages:</span> {result.pages ?? "—"}</div>
                    <div><span className="font-medium">Path:</span> {result.storage_path}</div>
                    <div><span className="font-medium">Created:</span> {result.created_at ?? "—"}</div>
                </div>
            )}
        </div>
    )
}