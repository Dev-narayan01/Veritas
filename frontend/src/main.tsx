import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'sonner'
import './styles/index.css'
import { App } from './App'
import { initApi } from './lib/api'
import { useUi } from './store'

async function bootstrap() {
  const state = await initApi((message) => {
    useUi.getState().setHealth(null, 'offline', message)
  })
  useUi.getState().setHealth(state.health, state.mode, state.message)
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <BrowserRouter>
        <App />
        <Toaster theme="dark" position="bottom-right" richColors />
      </BrowserRouter>
    </React.StrictMode>,
  )
}

void bootstrap()
