import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
    plugins: [react(), tailwindcss()],
    build: {
        // FastAPI serves the built app from api/static
        outDir: '../api/static',
        emptyOutDir: true,
    },
    server: {
        // Dev only: forward API and file requests to uvicorn
        proxy: {
            '/api': 'http://localhost:8000',
            '/files': 'http://localhost:8000',
        },
    },
})
