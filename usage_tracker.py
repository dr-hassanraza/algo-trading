import user_auth

USAGE_LIMIT = 10

def check_usage(username: str) -> bool:
    """Checks if a user has exceeded their usage limit."""
    user_data = user_auth.get_user_data(username)
    return user_data.get("usage_count", 0) < USAGE_LIMIT

def record_usage(username: str):
    """Increments the usage count for a user."""
    user_auth.increment_usage(username)

def get_remaining_usage(username: str) -> int:
    """Gets the remaining usage for a user."""
    user_data = user_auth.get_user_data(username)
    return max(0, USAGE_LIMIT - user_data.get("usage_count", 0))
