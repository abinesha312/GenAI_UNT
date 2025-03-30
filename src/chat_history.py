import redis
import json
import os
from typing import List, Dict, Optional
from langchain_redis import RedisChatMessageHistory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Redis connection string
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Message history expiration (in seconds)
MESSAGE_TTL = 60 * 60 * 24 * 7  # 7 days

def get_chat_history(user_id: str, chat_id: str, max_messages: int = 10) -> RedisChatMessageHistory:
    """Get chat history for a specific user and chat session."""
    session_id = f"{user_id}:{chat_id}"
    
    # Initialize Redis chat history with TTL
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        ttl=MESSAGE_TTL
    )
    
    return history

def get_user_chat_sessions(user_id: str) -> List[str]:
    """Get all chat sessions for a user."""
    # Connect to Redis
    redis_client = redis.from_url(REDIS_URL)
    
    # Get all keys matching the user's pattern
    pattern = f"chat:{user_id}:*"
    keys = redis_client.keys(pattern)
    
    # Extract chat IDs from keys
    chat_ids = [key.split(":")[-1] for key in keys]
    return chat_ids

def get_recent_chats(user_id: str, limit: int = 5) -> List[Dict]:
    """Get recent chat sessions with their first and last messages."""
    redis_client = redis.from_url(REDIS_URL)
    chats = []
    
    for chat_id in get_user_chat_sessions(user_id)[:limit]:
        history = get_chat_history(user_id, chat_id)
        messages = history.messages
        
        if messages:
            # Get first and last message
            first_message = messages[0].content if messages else ""
            last_message = messages[-1].content if messages else ""
            
            # Get timestamp if available
            timestamp = redis_client.hget(f"chat_meta:{user_id}:{chat_id}", "timestamp") or "Unknown"
            
            chats.append({
                "chat_id": chat_id,
                "first_message": first_message[:50] + "..." if len(first_message) > 50 else first_message,
                "last_message": last_message[:50] + "..." if len(last_message) > 50 else last_message,
                "timestamp": timestamp,
                "message_count": len(messages)
            })
    
    return chats
