import chainlit as cl
from typing import Optional, Dict
import uuid
from auth_utils import create_user, get_user, verify_password

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """
    Authenticate a user with username and password.
    """
    # Get user from Redis
    user_data = get_user(username)
    
    if user_data and verify_password(user_data["password"], password):
        # Create user object with appropriate role
        return cl.User(
            identifier=username,
            metadata={
                "role": user_data.get("role", "user"),
                "email": user_data.get("email", ""),
            }
        )
    return None

@cl.on_settings_update
async def setup_settings_callback(settings: Dict):
    """Handle registration through settings."""
    if "register" in settings:
        register_data = settings["register"]
        username = register_data.get("username", "")
        password = register_data.get("password", "")
        email = register_data.get("email", "")
        
        if not (username and password):
            return {"success": False, "error": "Username and password are required"}
        
        # Create new user
        success = create_user(username, password, email)
        
        if success:
            return {"success": True, "message": "User registered successfully"}
        else:
            return {"success": False, "error": "Username already exists"}
    
    return {"success": False, "error": "Invalid action"}
