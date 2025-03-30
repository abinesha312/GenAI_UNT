import bcrypt
import redis
import os
from typing import Optional, Dict
import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    password=os.getenv("REDIS_PASSWORD", None),
    decode_responses=True
)

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)  # Higher rounds = more secure but slower
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(provided_password.encode(), stored_password.encode())

def user_exists(username: str) -> bool:
    """Check if a user exists in Redis."""
    return redis_client.exists(f"user:{username}")

def create_user(username: str, password: str, email: str = "", role: str = "user") -> bool:
    """
    Create a new user in Redis.
    Returns True if successful, False if username already exists.
    """
    if user_exists(username):
        return False
    
    hashed_password = hash_password(password)
    redis_client.hset(f"user:{username}", mapping={
        "password": hashed_password,
        "email": email,
        "role": role,
        "created_at": redis_client.time()[0]  # Current timestamp
    })
    return True

def get_user(username: str) -> Optional[Dict]:
    """Get user data from Redis."""
    if not user_exists(username):
        return None
    
    user_data = redis_client.hgetall(f"user:{username}")
    return user_data
