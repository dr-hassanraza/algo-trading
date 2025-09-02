import user_auth

USAGE_LIMIT = 10

def check_usage(username: str) -> bool:
    """Checks if a user has exceeded their usage limit."""
    try:
        user_data = user_auth.get_user_data(username)
        usage_count = user_data.get("usage_count", 0)
        # Ensure usage_count is a number
        if not isinstance(usage_count, (int, float)):
            usage_count = 0
        return usage_count < USAGE_LIMIT
    except Exception as e:
        # If there's any error, allow usage (fail open)
        print(f"Usage check error: {e}")
        return True

def record_usage(username: str):
    """Increments the usage count for a user."""
    try:
        user_auth.increment_usage(username)
    except Exception as e:
        # If recording fails, log but don't crash
        print(f"Usage recording error: {e}")
        pass

def get_remaining_usage(username: str) -> int:
    """Gets the remaining usage for a user."""
    try:
        user_data = user_auth.get_user_data(username)
        usage_count = user_data.get("usage_count", 0)
        # Ensure usage_count is a number
        if not isinstance(usage_count, (int, float)):
            usage_count = 0
        return max(0, USAGE_LIMIT - usage_count)
    except Exception as e:
        # If there's an error, return full limit
        print(f"Usage remaining check error: {e}")
        return USAGE_LIMIT