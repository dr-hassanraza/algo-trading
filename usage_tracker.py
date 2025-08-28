
import json
from datetime import datetime
from pathlib import Path

USAGE_FILE = Path("usage.json")
DAILY_LIMIT = 10

def get_usage() -> dict:
    """Reads the usage.json file and returns a dictionary of usage data."""
    if not USAGE_FILE.exists():
        return {}
    
    with open(USAGE_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_usage(usage: dict):
    """Saves the usage dictionary to the usage.json file."""
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f, indent=4)

def check_usage(username: str) -> bool:
    """Checks if a user has exceeded their daily usage limit."""
    usage = get_usage()
    today = datetime.now().strftime("%Y-%m-%d")
    
    user_usage = usage.get(username, {})
    
    if user_usage.get("date") != today:
        return True  # New day, so usage is allowed
    
    return user_usage.get("count", 0) < DAILY_LIMIT

def record_usage(username: str):
    """Increments the usage count for a user."""
    usage = get_usage()
    today = datetime.now().strftime("%Y-%m-%d")
    
    user_usage = usage.get(username, {})
    
    if user_usage.get("date") == today:
        user_usage["count"] += 1
    else:
        user_usage["date"] = today
        user_usage["count"] = 1
        
    usage[username] = user_usage
    save_usage(usage)

def get_remaining_usage(username: str) -> int:
    """Gets the remaining usage for a user."""
    usage = get_usage()
    today = datetime.now().strftime("%Y-%m-%d")
    
    user_usage = usage.get(username, {})
    
    if user_usage.get("date") != today:
        return DAILY_LIMIT
        
    return max(0, DAILY_LIMIT - user_usage.get("count", 0))
