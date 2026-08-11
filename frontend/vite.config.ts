import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/verify': 'http://localhost:8000',
      '/verify-text': 'http://localhost:8000',
      '/history': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/chats': 'http://localhost:8000',
      '/api': 'http://localhost:8000',   // TTS narration: /api/tts
    }
  }
})
