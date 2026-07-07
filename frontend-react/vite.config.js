import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ticket':     'http://localhost:8000',
      '/health':     'http://localhost:8000',
      '/escalations':'http://localhost:8000',
      '/docs':       'http://localhost:8000',
    },
  },
})
