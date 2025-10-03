import UploadBox from "./components/UploadBox"
import DocumentList from "./components/DocumentList"
import { useEffect, useState } from "react"

export default function App() {
    const [status, setStatus] = useState("checking...")

    useEffect(() => {
        fetch("/api/status")
            .then((r) => r.json())
            .then((d) => setStatus(d.status))
            .catch(() => setStatus("down"))
    }, [])

    return (
        <main className="min-h-screen w-screen bg-gray-50 text-gray-900">
            <div className="mx-auto max-w-3xl p-6">
                <header className="mb-6">
                    <h1 className="text-3xl font-bold">ParseIQ</h1>
                    <p className="text-sm text-gray-600">Your everyday life-admin copilot</p>
                </header>

                <section className="mb-6 rounded-2xl border bg-white p-6 shadow-sm">
                    <h2 className="mb-2 text-xl font-semibold">API status</h2>
                    <div className="inline-block rounded-full bg-gray-100 px-3 py-1 text-sm">
                        {status}
                    </div>
                </section>

                <UploadBox />
                <DocumentList />
            </div>
        </main>
    )
}