import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { Auth0Provider } from '@auth0/auth0-react'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Auth0Provider
      domain="dev-lwrdg131t817pkv7.us.auth0.com"
      clientId="Vgo5Hf8Clp2nodLdest3mqEqWg8pnQ4U"
      authorizationParams={{
        redirect_uri: window.location.origin,
        audience: 'https://support-resolution-api/',
        scope: 'openid profile email',
      }}
    >
      <App />
    </Auth0Provider>
  </StrictMode>,
)
