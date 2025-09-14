"""
🔐 Secure OAuth2 Authentication System
=====================================

Implements secure Google and LinkedIn OAuth2 authentication
to replace the insecure placeholder login system.

Features:
- JWT token management
- Secure session handling
- OAuth2 provider integration
- User profile management
"""

import streamlit as st
import requests
import jwt
import json
import hashlib
import secrets
import base64
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from urllib.parse import urlencode, parse_qs, urlparse
import logging

logger = logging.getLogger(__name__)

class OAuth2Config:
    """OAuth2 configuration management"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load OAuth2 configuration from secrets or environment"""
        try:
            # Try Streamlit secrets first (production)
            if hasattr(st, 'secrets') and 'google' in st.secrets:
                return {
                    'google': {
                        'client_id': st.secrets['google']['client_id'],
                        'client_secret': st.secrets['google']['client_secret'],
                        'redirect_uri': st.secrets['google'].get('redirect_uri', 'http://localhost:8501'),
                        'scope': 'openid email profile',
                        'auth_url': 'https://accounts.google.com/o/oauth2/v2/auth',
                        'token_url': 'https://oauth2.googleapis.com/token',
                        'userinfo_url': 'https://www.googleapis.com/oauth2/v2/userinfo'
                    },
                    'linkedin': {
                        'client_id': st.secrets.get('linkedin', {}).get('client_id', ''),
                        'client_secret': st.secrets.get('linkedin', {}).get('client_secret', ''),
                        'redirect_uri': st.secrets.get('linkedin', {}).get('redirect_uri', 'http://localhost:8501'),
                        'scope': 'r_liteprofile r_emailaddress',
                        'auth_url': 'https://www.linkedin.com/oauth/v2/authorization',
                        'token_url': 'https://www.linkedin.com/oauth/v2/accessToken',
                        'userinfo_url': 'https://api.linkedin.com/v2/me'
                    },
                    'jwt_secret': st.secrets.get('auth', {}).get('secret_key', self._generate_fallback_secret()),
                    'jwt_algorithm': 'HS256',
                    'token_expire_minutes': 1440  # 24 hours
                }
            else:
                # Fallback configuration for development
                return self._get_fallback_config()
                
        except Exception as e:
            logger.warning(f"Failed to load OAuth2 config: {e}")
            return self._get_fallback_config()
    
    def _generate_fallback_secret(self) -> str:
        """Generate a secure fallback secret"""
        return secrets.token_urlsafe(32)
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback configuration for development"""
        return {
            'google': {
                'client_id': '',
                'client_secret': '',
                'redirect_uri': 'http://localhost:8501',
                'scope': 'openid email profile',
                'auth_url': 'https://accounts.google.com/o/oauth2/v2/auth',
                'token_url': 'https://oauth2.googleapis.com/token',
                'userinfo_url': 'https://www.googleapis.com/oauth2/v2/userinfo'
            },
            'linkedin': {
                'client_id': '',
                'client_secret': '',
                'redirect_uri': 'http://localhost:8501',
                'scope': 'r_liteprofile r_emailaddress',
                'auth_url': 'https://www.linkedin.com/oauth/v2/authorization',
                'token_url': 'https://www.linkedin.com/oauth/v2/accessToken',
                'userinfo_url': 'https://api.linkedin.com/v2/me'
            },
            'jwt_secret': self._generate_fallback_secret(),
            'jwt_algorithm': 'HS256',
            'token_expire_minutes': 1440
        }
    
    def is_configured(self, provider: str) -> bool:
        """Check if OAuth2 provider is properly configured"""
        config = self.config.get(provider, {})
        return bool(config.get('client_id') and config.get('client_secret'))
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific provider"""
        return self.config.get(provider, {})

class JWTManager:
    """JWT token management"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, user_data: Dict) -> str:
        """Create a JWT token for authenticated user"""
        payload = {
            'user_id': user_data.get('id', ''),
            'email': user_data.get('email', ''),
            'name': user_data.get('name', ''),
            'picture': user_data.get('picture', ''),
            'provider': user_data.get('provider', ''),
            'exp': datetime.utcnow() + timedelta(minutes=1440),  # 24 hours
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None

class OAuth2Provider:
    """Base OAuth2 provider class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_state_key = f"oauth2_{config.get('name', 'unknown')}"
    
    def generate_auth_url(self) -> Tuple[str, str]:
        """Generate OAuth2 authorization URL and state"""
        state = secrets.token_urlsafe(32)
        
        params = {
            'client_id': self.config['client_id'],
            'redirect_uri': self.config['redirect_uri'],
            'scope': self.config['scope'],
            'response_type': 'code',
            'state': state,
        }
        
        # Add provider-specific parameters
        if 'google' in self.config.get('auth_url', ''):
            params['access_type'] = 'offline'
            params['prompt'] = 'consent'
        
        auth_url = f"{self.config['auth_url']}?{urlencode(params)}"
        return auth_url, state
    
    def exchange_code_for_token(self, code: str) -> Optional[str]:
        """Exchange authorization code for access token"""
        try:
            data = {
                'client_id': self.config['client_id'],
                'client_secret': self.config['client_secret'],
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': self.config['redirect_uri']
            }
            
            response = requests.post(
                self.config['token_url'],
                data=data,
                headers={'Accept': 'application/json'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                return token_data.get('access_token')
            else:
                logger.error(f"Token exchange failed: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return None
    
    def get_user_info(self, access_token: str) -> Optional[Dict]:
        """Get user information using access token"""
        try:
            headers = {'Authorization': f'Bearer {access_token}'}
            response = requests.get(self.config['userinfo_url'], headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"User info fetch failed: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"User info error: {e}")
            return None

class GoogleOAuth2(OAuth2Provider):
    """Google OAuth2 implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        google_config = config.copy()
        google_config['name'] = 'google'
        super().__init__(google_config)
    
    def normalize_user_data(self, user_info: Dict) -> Dict:
        """Normalize Google user data"""
        return {
            'id': user_info.get('id', ''),
            'email': user_info.get('email', ''),
            'name': user_info.get('name', ''),
            'picture': user_info.get('picture', ''),
            'provider': 'google',
            'verified_email': user_info.get('verified_email', False)
        }

class LinkedInOAuth2(OAuth2Provider):
    """LinkedIn OAuth2 implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        linkedin_config = config.copy()
        linkedin_config['name'] = 'linkedin'
        super().__init__(linkedin_config)
    
    def get_user_info(self, access_token: str) -> Optional[Dict]:
        """Get LinkedIn user information (requires multiple API calls)"""
        try:
            headers = {'Authorization': f'Bearer {access_token}'}
            
            # Get basic profile
            profile_response = requests.get(
                'https://api.linkedin.com/v2/me?projection=(id,firstName,lastName,profilePicture(displayImage~:playableStreams))',
                headers=headers
            )
            
            # Get email address
            email_response = requests.get(
                'https://api.linkedin.com/v2/emailAddress?q=members&projection=(elements*(handle~))',
                headers=headers
            )
            
            if profile_response.status_code == 200:
                profile_data = profile_response.json()
                email_data = email_response.json() if email_response.status_code == 200 else {}
                
                # Combine data
                user_info = {
                    'id': profile_data.get('id', ''),
                    'firstName': profile_data.get('firstName', {}).get('localized', {}).get('en_US', ''),
                    'lastName': profile_data.get('lastName', {}).get('localized', {}).get('en_US', ''),
                }
                
                # Extract email
                if 'elements' in email_data and email_data['elements']:
                    user_info['email'] = email_data['elements'][0].get('handle~', {}).get('emailAddress', '')
                
                # Extract profile picture
                profile_pic = profile_data.get('profilePicture', {}).get('displayImage~', {})
                if 'elements' in profile_pic and profile_pic['elements']:
                    user_info['picture'] = profile_pic['elements'][0].get('identifiers', [{}])[0].get('identifier', '')
                
                return user_info
            else:
                logger.error(f"LinkedIn profile fetch failed: {profile_response.text}")
                return None
                
        except Exception as e:
            logger.error(f"LinkedIn user info error: {e}")
            return None
    
    def normalize_user_data(self, user_info: Dict) -> Dict:
        """Normalize LinkedIn user data"""
        first_name = user_info.get('firstName', '')
        last_name = user_info.get('lastName', '')
        full_name = f"{first_name} {last_name}".strip()
        
        return {
            'id': user_info.get('id', ''),
            'email': user_info.get('email', ''),
            'name': full_name or 'LinkedIn User',
            'picture': user_info.get('picture', ''),
            'provider': 'linkedin',
            'verified_email': bool(user_info.get('email'))
        }

class SecureOAuth2Manager:
    """Secure OAuth2 authentication manager"""
    
    def __init__(self):
        self.config = OAuth2Config()
        self.jwt_manager = JWTManager(self.config.config['jwt_secret'])
        self.google_oauth = GoogleOAuth2(self.config.get_provider_config('google'))
        self.linkedin_oauth = LinkedInOAuth2(self.config.get_provider_config('linkedin'))
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if OAuth2 provider is available and configured"""
        return self.config.is_configured(provider)
    
    def get_auth_url(self, provider: str) -> Optional[Tuple[str, str]]:
        """Get authorization URL for provider"""
        try:
            if provider == 'google' and self.is_provider_available('google'):
                return self.google_oauth.generate_auth_url()
            elif provider == 'linkedin' and self.is_provider_available('linkedin'):
                return self.linkedin_oauth.generate_auth_url()
            else:
                return None
        except Exception as e:
            logger.error(f"Auth URL generation error for {provider}: {e}")
            return None
    
    def handle_callback(self, provider: str, code: str, state: str) -> Optional[Dict]:
        """Handle OAuth2 callback and authenticate user"""
        try:
            # Verify state (CSRF protection)
            stored_state = st.session_state.get(f'oauth2_state_{provider}')
            if not stored_state or stored_state != state:
                logger.error(f"OAuth2 state mismatch for {provider}")
                return None
            
            # Get the appropriate OAuth2 provider
            oauth_provider = None
            if provider == 'google':
                oauth_provider = self.google_oauth
            elif provider == 'linkedin':
                oauth_provider = self.linkedin_oauth
            
            if not oauth_provider:
                return None
            
            # Exchange code for token
            access_token = oauth_provider.exchange_code_for_token(code)
            if not access_token:
                return None
            
            # Get user info
            user_info = oauth_provider.get_user_info(access_token)
            if not user_info:
                return None
            
            # Normalize user data
            user_data = oauth_provider.normalize_user_data(user_info)
            
            # Create JWT token
            jwt_token = self.jwt_manager.create_token(user_data)
            
            # Store in session
            st.session_state['oauth2_token'] = jwt_token
            st.session_state['authenticated'] = True
            st.session_state['username'] = user_data['name']
            st.session_state['user_email'] = user_data['email']
            st.session_state['user_picture'] = user_data['picture']
            st.session_state['auth_provider'] = provider
            st.session_state['login_time'] = datetime.now().isoformat()
            
            # Clean up OAuth2 state
            if f'oauth2_state_{provider}' in st.session_state:
                del st.session_state[f'oauth2_state_{provider}']
            
            return user_data
            
        except Exception as e:
            logger.error(f"OAuth2 callback error for {provider}: {e}")
            return None
    
    def verify_session(self) -> bool:
        """Verify current OAuth2 session"""
        try:
            token = st.session_state.get('oauth2_token')
            if not token:
                return False
            
            payload = self.jwt_manager.verify_token(token)
            if not payload:
                self.clear_session()
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Session verification error: {e}")
            self.clear_session()
            return False
    
    def clear_session(self):
        """Clear OAuth2 session"""
        oauth_keys = [
            'oauth2_token', 'authenticated', 'username', 'user_email',
            'user_picture', 'auth_provider', 'login_time'
        ]
        
        for key in oauth_keys:
            if key in st.session_state:
                del st.session_state[key]
    
    def get_user_info_from_session(self) -> Optional[Dict]:
        """Get user info from current session"""
        if not self.verify_session():
            return None
        
        token = st.session_state.get('oauth2_token')
        if token:
            return self.jwt_manager.verify_token(token)
        return None

# Global instance
oauth_manager = SecureOAuth2Manager()

def get_oauth_manager() -> SecureOAuth2Manager:
    """Get the global OAuth2 manager instance"""
    return oauth_manager