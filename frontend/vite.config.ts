import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the FastAPI server runs on :8000; in production it serves
// the built files itself, so the app always talks to a same-origin /api.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8000', ws: true },
    },
  },
})
