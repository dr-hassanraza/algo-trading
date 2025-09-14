"""
🔐 Secure OAuth2 Streamlit Components
=====================================

Custom Streamlit components for secure OAuth2 authentication
"""

import streamlit as st
import streamlit.components.v1 as components
from urllib.parse import quote, urlencode, parse_qs, urlparse
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

def oauth2_button(provider: str, config: Dict[str, Any], button_text: str = None, key: str = None) -> bool:
    """
    Create a secure OAuth2 login button
    
    Args:
        provider: OAuth2 provider name ('google' or 'linkedin')
        config: Provider configuration dictionary
        button_text: Custom button text (optional)
        key: Unique key for the button (optional)
    
    Returns:
        bool: True if button was clicked, False otherwise
    """
    
    if not button_text:
        button_text = f"🔐 Login with {provider.title()}"
    
    if not key:
        key = f"oauth2_button_{provider}"
    
    # Check if provider is configured
    if not config.get('client_id') or not config.get('client_secret'):
        st.error(f"❌ {provider.title()} OAuth2 is not configured. Check OAUTH_SETUP.md for instructions.")
        return False
    
    # Create button
    if st.button(button_text, key=key, type="secondary", use_container_width=True):
        # Generate OAuth2 authorization URL
        from src.oauth_auth import get_oauth_manager
        oauth_manager = get_oauth_manager()
        
        auth_result = oauth_manager.get_auth_url(provider)
        if auth_result:
            auth_url, state = auth_result
            
            # Store state in session for CSRF protection
            st.session_state[f'oauth2_state_{provider}'] = state
            
            # Create redirect HTML
            redirect_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Redirecting to {provider.title()}...</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-align: center;
                    }}
                    .container {{
                        background: rgba(255, 255, 255, 0.1);
                        padding: 2rem;
                        border-radius: 15px;
                        backdrop-filter: blur(10px);
                    }}
                    .spinner {{
                        border: 4px solid rgba(255,255,255,0.3);
                        border-radius: 50%;
                        border-top: 4px solid white;
                        width: 40px;
                        height: 40px;
                        animation: spin 1s linear infinite;
                        margin: 0 auto 1rem auto;
                    }}
                    @keyframes spin {{
                        0% {{ transform: rotate(0deg); }}
                        100% {{ transform: rotate(360deg); }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="spinner"></div>
                    <h2>Connecting to {provider.title()}...</h2>
                    <p>Please wait while we redirect you to {provider.title()} for secure authentication.</p>
                </div>
                <script>
                    window.location.href = "{auth_url}";
                </script>
            </body>
            </html>
            """
            
            # Display redirect page
            components.html(redirect_html, height=400)
            return True
        else:
            st.error(f"❌ Failed to generate {provider.title()} authentication URL")
            return False
    
    return False

def handle_oauth2_callback():
    """Handle OAuth2 callback from URL parameters"""
    try:
        # Check if we have OAuth2 callback parameters
        query_params = st.experimental_get_query_params()
        
        if 'code' in query_params and 'state' in query_params:
            code = query_params['code'][0]
            state = query_params['state'][0]
            
            # Determine provider from state or session
            provider = None
            for p in ['google', 'linkedin']:
                if st.session_state.get(f'oauth2_state_{p}') == state:
                    provider = p
                    break
            
            if provider:
                from src.oauth_auth import get_oauth_manager
                oauth_manager = get_oauth_manager()
                
                # Handle the callback
                user_data = oauth_manager.handle_callback(provider, code, state)
                
                if user_data:
                    st.success(f"✅ Successfully authenticated with {provider.title()}!")
                    st.balloons()
                    
                    # Clear query parameters
                    st.experimental_set_query_params()
                    
                    # Rerun to refresh the app
                    st.rerun()
                else:
                    st.error(f"❌ Authentication with {provider.title()} failed. Please try again.")
                    st.experimental_set_query_params()
            else:
                st.error("❌ Invalid OAuth2 callback state")
                st.experimental_set_query_params()
    
    except Exception as e:
        logger.error(f"OAuth2 callback handling error: {e}")
        st.error("❌ OAuth2 authentication error. Please try again.")

def render_oauth2_login_section(title: str = "🔐 Secure Social Login"):
    """
    Render a complete OAuth2 login section
    
    Args:
        title: Section title
    """
    
    from src.oauth_auth import get_oauth_manager
    oauth_manager = get_oauth_manager()
    
    st.markdown(f"### {title}")
    
    # Check for OAuth2 callback first
    handle_oauth2_callback()
    
    # Show login options only if not authenticated
    if not st.session_state.get('authenticated', False):
        
        st.markdown("""
        <div style='background-color: #e8f4fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom: 1rem;'>
            <p style='margin: 0; color: #1f77b4;'>
                🔒 <strong>Secure Authentication</strong><br>
                Choose your preferred method to sign in securely. We use industry-standard OAuth2 for your protection.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Google OAuth2 login
            google_config = oauth_manager.config.get_provider_config('google')
            if oauth_manager.is_provider_available('google'):
                oauth2_button('google', google_config, "🔍 Continue with Google", "google_oauth_btn")
            else:
                st.info("🔍 **Google Login**: Setup required (see OAUTH_SETUP.md)")
        
        with col2:
            # LinkedIn OAuth2 login
            linkedin_config = oauth_manager.config.get_provider_config('linkedin')
            if oauth_manager.is_provider_available('linkedin'):
                oauth2_button('linkedin', linkedin_config, "💼 Continue with LinkedIn", "linkedin_oauth_btn")
            else:
                st.info("💼 **LinkedIn Login**: Setup required (see OAUTH_SETUP.md)")
        
        # Configuration status
        if not oauth_manager.is_provider_available('google') and not oauth_manager.is_provider_available('linkedin'):
            st.warning("""
            ⚠️ **OAuth2 Setup Required**
            
            To enable secure social login, please:
            1. Follow the setup guide in `OAUTH_SETUP.md`
            2. Configure your Google and/or LinkedIn OAuth2 credentials
            3. Add them to `.streamlit/secrets.toml`
            """)
            
            with st.expander("📖 Quick Setup Guide", expanded=False):
                st.markdown("""
                **Google Setup:**
                1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                2. Create a project and enable Google+ API
                3. Create OAuth2 credentials
                4. Add `http://localhost:8501` as redirect URI
                
                **LinkedIn Setup:**
                1. Go to [LinkedIn Developer Portal](https://www.linkedin.com/developers/)
                2. Create an app
                3. Configure OAuth2 settings
                4. Add `http://localhost:8501` as redirect URI
                
                **Configuration File:**
                Create `.streamlit/secrets.toml`:
                ```toml
                [google]
                client_id = "your-google-client-id"
                client_secret = "your-google-client-secret"
                
                [linkedin]
                client_id = "your-linkedin-client-id"  
                client_secret = "your-linkedin-client-secret"
                
                [auth]
                secret_key = "your-jwt-secret-key"
                ```
                """)
    
    else:
        # Show authenticated user info
        render_oauth2_user_info()

def render_oauth2_user_info():
    """Render authenticated user information"""
    
    from src.oauth_auth import get_oauth_manager
    oauth_manager = get_oauth_manager()
    
    user_info = oauth_manager.get_user_info_from_session()
    
    if user_info:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            # User profile card
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 1rem 0;'>
            """, unsafe_allow_html=True)
            
            # User avatar
            if user_info.get('picture'):
                st.image(user_info['picture'], width=80)
            else:
                st.markdown("👤", unsafe_allow_html=True)
            
            st.markdown(f"**{user_info.get('name', 'User')}**")
            st.markdown(f"📧 {user_info.get('email', 'No email')}")
            
            provider = user_info.get('provider', 'unknown').title()
            st.markdown(f"🔐 Authenticated via {provider}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Logout button
            if st.button("🚪 Logout", key="oauth2_logout", type="secondary"):
                oauth_manager.clear_session()
                st.success("✅ Logged out successfully")
                st.rerun()

def oauth2_login_sidebar():
    """Render OAuth2 login in sidebar"""
    
    from src.oauth_auth import get_oauth_manager
    oauth_manager = get_oauth_manager()
    
    if not st.session_state.get('authenticated', False):
        st.sidebar.markdown("### 🔐 Login")
        
        # Quick OAuth2 buttons in sidebar
        google_config = oauth_manager.config.get_provider_config('google')
        if oauth_manager.is_provider_available('google'):
            if st.sidebar.button("🔍 Google", key="sidebar_google", use_container_width=True):
                oauth2_button('google', google_config)
        
        linkedin_config = oauth_manager.config.get_provider_config('linkedin')  
        if oauth_manager.is_provider_available('linkedin'):
            if st.sidebar.button("💼 LinkedIn", key="sidebar_linkedin", use_container_width=True):
                oauth2_button('linkedin', linkedin_config)
    
    else:
        # Show user info in sidebar
        user_info = oauth_manager.get_user_info_from_session()
        if user_info:
            st.sidebar.markdown("### 👤 Profile")
            
            if user_info.get('picture'):
                st.sidebar.image(user_info['picture'], width=60)
            
            st.sidebar.write(f"**{user_info.get('name', 'User')}**")
            st.sidebar.write(f"📧 {user_info.get('email', '')}")
            
            if st.sidebar.button("🚪 Logout", key="sidebar_logout"):
                oauth_manager.clear_session()
                st.rerun()

def check_oauth2_authentication() -> bool:
    """
    Check if user is authenticated via OAuth2
    
    Returns:
        bool: True if authenticated, False otherwise
    """
    
    from src.oauth_auth import get_oauth_manager
    oauth_manager = get_oauth_manager()
    
    # Handle OAuth2 callback if present
    handle_oauth2_callback()
    
    # Verify session
    return oauth_manager.verify_session()