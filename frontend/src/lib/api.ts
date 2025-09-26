// ------------------------------------------
// Interface for API response after upload
// ------------------------------------------
export type UploadResponse = {
    id: number
    title: string
    media_type: "pdf" | "image" | "text" | "audio"
    pages: number | null
    storage_path: string
    created_at: string
}


// -----------------------------------------------
// Function to upload a document to the backend
// -----------------------------------------------
export async function uploadDocument(file: File): Promise<UploadResponse> {
    // Create a FormData object to hold the file
    const fd = new FormData()
    fd.append("file", file)

    // Send a POST request to the upload endpoint
    const res = await fetch("/api/documents/upload", {
        method: "POST",
        body: fd,
    })

    // If the response is not OK, throw an error with the response message
    if (!res.ok) {
        const msg = await res.text().catch(() => "")
        throw new Error(msg || `Upload failed with ${res.status}`)
    }

    // Parse and return the JSON response
    return res.json()
}