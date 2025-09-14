# 🔐 OAuth2 Social Login Setup Guide

## Overview

This guide will help you set up secure Google and LinkedIn OAuth2 authentication for the trading system, replacing the current placeholder login system.

## 🚀 Quick Setup

### Step 1: Install Required Dependencies

```bash
# Install OAuth2 libraries
pip install streamlit-oauth
pip install google-auth-oauthlib
pip install requests-oauthlib

# Alternative OAuth library (backup option)
pip install streamlit-authenticator
pip install python-jose[cryptography]
```

### Step 2: Get OAuth2 Credentials

#### 🔵 Google Cloud Console Setup

1. **Go to Google Cloud Console**
   - Visit: https://console.cloud.google.com/
   - Sign in with your Google account

2. **Create a New Project**
   - Click "Select a project" → "New Project"
   - Name: "PSX-Algo-Trading" (or your preferred name)
   - Click "Create"

3. **Enable Google+ API**
   - Go to "APIs & Services" → "Library"
   - Search for "Google+ API" and enable it
   - Also enable "Google Identity" API

4. **Create OAuth2 Credentials**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth 2.0 Client ID"
   - Application type: "Web application"
   - Name: "PSX Trading System"

5. **Configure Authorized Redirect URIs**
   ```
   # For local development:
   http://localhost:8501
   http://localhost:8501/component/streamlit_oauth.authorize_button/index.html
   
   # For production (replace with your domain):
   https://your-domain.com
   https://your-domain.com/component/streamlit_oauth.authorize_button/index.html
   ```

6. **Copy Your Credentials**
   - Copy the `Client ID` and `Client Secret`
   - Keep these secure!

#### 🔗 LinkedIn Developer Portal Setup

1. **Go to LinkedIn Developer Portal**
   - Visit: https://www.linkedin.com/developers/
   - Sign in with your LinkedIn account

2. **Create a New App**
   - Click "Create App"
   - App name: "PSX Algo Trading System"
   - LinkedIn Page: Create or select a LinkedIn page
   - App use: "Sign In with LinkedIn"

3. **Configure OAuth2 Settings**
   - Go to "Auth" tab
   - Add redirect URLs:
   ```
   http://localhost:8501
   http://localhost:8501/component/streamlit_oauth.authorize_button/index.html
   ```

4. **Request Permissions**
   - Under "OAuth 2.0 scopes", request:
     - `r_liteprofile` (basic profile info)
     - `r_emailaddress` (email address)

5. **Copy Your Credentials**
   - Copy the `Client ID` and `Client Secret`

### Step 3: Set Up Streamlit Secrets

Create a `.streamlit/secrets.toml` file in your project root:

```toml
# .streamlit/secrets.toml
[google]
client_id = "your-google-client-id.googleusercontent.com"
client_secret = "your-google-client-secret"
redirect_uri = "http://localhost:8501"

[linkedin]
client_id = "your-linkedin-client-id"
client_secret = "your-linkedin-client-secret"
redirect_uri = "http://localhost:8501"

[auth]
secret_key = "your-super-secret-jwt-key-at-least-32-characters-long"
algorithm = "HS256"
access_token_expire_minutes = 1440

[database]
# If using a database for user management
db_url = "your-database-url"
```

### Step 4: Secure Your Secrets

```bash
# Add secrets to .gitignore
echo ".streamlit/secrets.toml" >> .gitignore
echo ".env" >> .gitignore

# Create environment file as backup
cp .streamlit/secrets.toml .env.example
# Remove actual values from .env.example for sharing
```

## 🔧 Production Deployment

### For Streamlit Cloud

1. **Upload Secrets**
   - In Streamlit Cloud dashboard
   - Go to app settings → "Secrets"
   - Paste your secrets.toml content

2. **Update Redirect URIs**
   - Change redirect URIs to your Streamlit Cloud URL
   - Format: `https://your-app-name.streamlit.app`

### For Custom Domain

1. **Update OAuth2 Credentials**
   - Add your production domain to authorized redirect URIs
   - Update secrets.toml with production URLs

2. **SSL Certificate**
   - Ensure your domain has a valid SSL certificate
   - OAuth2 requires HTTPS in production

## 🔒 Security Best Practices

### 1. Environment-Specific Configuration

```python
# config.py
import streamlit as st
import os
from typing import Dict, Any

def get_oauth_config() -> Dict[str, Any]:
    """Get OAuth configuration based on environment"""
    
    # Try Streamlit secrets first (production)
    try:
        return {
            'google': {
                'client_id': st.secrets['google']['client_id'],
                'client_secret': st.secrets['google']['client_secret'],
                'redirect_uri': st.secrets['google']['redirect_uri'],
            },
            'linkedin': {
                'client_id': st.secrets['linkedin']['client_id'],
                'client_secret': st.secrets['linkedin']['client_secret'],
                'redirect_uri': st.secrets['linkedin']['redirect_uri'],
            },
            'jwt_secret': st.secrets['auth']['secret_key'],
        }
    except:
        # Fallback to environment variables (development)
        return {
            'google': {
                'client_id': os.getenv('GOOGLE_CLIENT_ID'),
                'client_secret': os.getenv('GOOGLE_CLIENT_SECRET'),
                'redirect_uri': os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501'),
            },
            'linkedin': {
                'client_id': os.getenv('LINKEDIN_CLIENT_ID'),
                'client_secret': os.getenv('LINKEDIN_CLIENT_SECRET'),
                'redirect_uri': os.getenv('LINKEDIN_REDIRECT_URI', 'http://localhost:8501'),
            },
            'jwt_secret': os.getenv('JWT_SECRET_KEY', 'your-fallback-secret-key'),
        }
```

### 2. Token Security

```python
# auth_utils.py
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict

class JWTHandler:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, user_data: Dict) -> str:
        """Create a JWT token"""
        payload = {
            'user_id': user_data.get('id'),
            'email': user_data.get('email'),
            'name': user_data.get('name'),
            'provider': user_data.get('provider'),
            'exp': datetime.utcnow() + timedelta(minutes=1440),  # 24 hours
            'iat': datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
```

## 🧪 Testing Your Setup

### Local Testing Checklist

- [ ] Google OAuth2 credentials configured
- [ ] LinkedIn OAuth2 credentials configured
- [ ] Redirect URIs match exactly
- [ ] Secrets file created and secured
- [ ] Dependencies installed
- [ ] App runs without errors
- [ ] OAuth2 login flow works
- [ ] User data is properly stored
- [ ] Session management works
- [ ] Logout functionality works

### Production Testing Checklist

- [ ] Production URLs configured
- [ ] SSL certificate valid
- [ ] Secrets uploaded to hosting platform
- [ ] OAuth2 flow works in production
- [ ] No sensitive data in logs
- [ ] Error handling works correctly
- [ ] Session persistence works
- [ ] Multiple users can login simultaneously

## 🚨 Troubleshooting

### Common Issues

1. **"redirect_uri_mismatch" Error**
   - Ensure redirect URI in code matches OAuth2 provider settings exactly
   - Check for trailing slashes, http vs https

2. **"invalid_client" Error**
   - Check Client ID and Client Secret are correct
   - Ensure OAuth2 app is properly configured

3. **Streamlit Component Not Loading**
   - Clear browser cache
   - Check network console for errors
   - Verify streamlit-oauth is installed correctly

4. **Secrets Not Found**
   - Check secrets.toml file location and format
   - Verify environment variables are set correctly
   - Check file permissions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In your OAuth2 implementation
import streamlit as st
st.write("Debug: OAuth2 config loaded", get_oauth_config())
```

## 📚 Additional Resources

- [Google OAuth2 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [LinkedIn OAuth2 Documentation](https://docs.microsoft.com/en-us/linkedin/shared/authentication/authorization-code-flow)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [JWT.io Debugger](https://jwt.io/) - Debug JWT tokens

---

🔐 **Security Note**: Never commit secrets to version control. Always use environment variables or secure secret management systems in production.