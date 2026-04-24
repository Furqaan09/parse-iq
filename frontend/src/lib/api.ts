// ------------------------------------------
// Interface for API response after upload
// ------------------------------------------
export type UploadResponse = {
    id: number
    title: string
    media_type: "pdf" | "image" | "text"
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



// ---------------------------------------------------
// Interface for a single document item in the list
// ---------------------------------------------------
export type DocumentItem = {
    id: number
    title: string
    media_type: "pdf" | "image" | "text"
    pages: number | null
    storage_path: string
}

// --------------------------------------------
// Interface for paginated list of documents
// --------------------------------------------
export type DocumentListResponse = {
    items: DocumentItem[]
    page: number
    page_size: number
    total: number
    has_next: boolean
}

// ------------------------------------------------------------------------
// Function to fetch a paginated list of documents with optional filters
// ------------------------------------------------------------------------
export async function listDocuments(params?: {
    page?: number
    page_size?: number
    media_type?: DocumentItem["media_type"]
    q?: string
}): Promise<DocumentListResponse> {
    const url = new URL("/api/documents", window.location.origin)
    if (params?.page) url.searchParams.set("page", String(params.page))
    if (params?.page_size) url.searchParams.set("page_size", String(params.page_size))
    if (params?.media_type) url.searchParams.set("media_type", params.media_type)
    if (params?.q) url.searchParams.set("q", params.q)

    const res = await fetch(url.toString().replace(window.location.origin, ""))
    if (!res.ok) {
        throw new Error(`Failed to list documents: ${res.status}`)
    }
    return res.json()
}



// -------------------------------------------
// Interface for detailed document response
// -------------------------------------------
export type GetDocumentResponse = {
    id: number
    title: string
    media_type: "pdf" | "image" | "text"
    pages: number | null
    storage_path: string
    file_url: string | null
}

// -------------------------------------------------------------------------
// Function to fetch detailed information about a specific document by ID
// -------------------------------------------------------------------------
export async function getDocument(id: number): Promise<GetDocumentResponse> {
    const res = await fetch(`/api/documents/${id}`)
    if (!res.ok) {
        throw new Error(`Failed to get document ${id}: ${res.status}`)
    }
    return res.json()
}


// -------------------------------------------------------
// Interface for a single citation in the chat response
// -------------------------------------------------------
export type ChatCitation = {
    n: number
    document_id: number
    title: string
    page: number | null
    file_url: string | null
    snippet: string
}

// --------------------------------------------------------
// Interface for the response from the chat ask endpoint
// --------------------------------------------------------
export type ChatAskResponse = {
    answer: string
    citations: ChatCitation[]
}

// -----------------------------------------------------------------------
// Function to send a chat message to the backend and receive an answer
// -----------------------------------------------------------------------
export async function askChat(params: {
    message: string
    top_k?: number
    document_ids?: number[]
}): Promise<ChatAskResponse> {
    const res = await fetch("/api/chat/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            message: params.message,
            top_k: params.top_k ?? 6,
            document_ids: params.document_ids ?? null,
        }),
    })

    if (!res.ok) {
        const msg = await res.text().catch(() => "")
        throw new Error(msg || `Chat request failed with ${res.status}`)
    }

    return res.json()
}