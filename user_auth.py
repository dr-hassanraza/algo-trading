import json
import hashlib
import os
from pathlib import Path

USERS_FILE = Path("users.json")

def hash_password(password: str, salt: bytes = None) -> str:
    """Hashes a password with a salt."""
    if salt is None:
        salt = os.urandom(16)
    
    salted_password = salt + password.encode('utf-8')
    hashed_password = hashlib.sha256(salted_password).hexdigest()
    
    return f"{salt.hex()}:{hashed_password}"

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verifies a provided password against a stored hash."""
    try:
        salt_hex, hashed_password = stored_password.split(':')
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
        
    return stored_password == hash_password(provided_password, salt)

def get_users() -> dict:
    """Reads the users.json file and returns a dictionary of users."""
    if not USERS_FILE.exists():
        return {}
    
    try:
        with open(USERS_FILE, "r") as f:
            try:
                users_data = json.load(f)
                # Ensure it's a dictionary
                if not isinstance(users_data, dict):
                    return {}
                return users_data
            except json.JSONDecodeError:
                # If JSON is corrupted, return empty dict
                return {}
    except (IOError, OSError):
        # If file can't be read, return empty dict
        return {}

def save_users(users: dict):
    """Saves the users dictionary to the users.json file."""
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)
    except (IOError, OSError) as e:
        # If we can't save, at least log the error (but don't crash the app)
        print(f"Warning: Could not save users file: {e}")
        pass

def add_user(username: str, password: str, name: str, email: str) -> bool:
    """Adds a new user to the users.json file."""
    users = get_users()
    
    if username in users:
        return False  # User already exists
    
    users[username] = {
        "password": hash_password(password),
        "name": name,
        "email": email,
        "usage_count": 0,
        "ip_address": ""
    }
    
    save_users(users)
    return True

def authenticate_user(username: str, password: str) -> bool:
    """Authenticates a user."""
    users = get_users()
    
    if username not in users:
        return False
    
    stored_password = users[username].get("password")
    if not stored_password:
        return False
        
    return verify_password(stored_password, password)

def update_user_ip(username: str, ip_address: str):
    """Updates the user's IP address."""
    users = get_users()
    if username in users:
        users[username]["ip_address"] = ip_address
        save_users(users)

def get_user_data(username: str) -> dict:
    """Retrieves all data for a specific user."""
    users = get_users()
    user_data = users.get(username, {})
    
    # Ensure all required fields exist with default values
    default_user = {
        "password": "",
        "name": username,
        "email": "",
        "usage_count": 0,
        "ip_address": ""
    }
    
    # Merge user data with defaults, ensuring all fields are present
    for key, default_value in default_user.items():
        if key not in user_data:
            user_data[key] = default_value
    
    return user_data

def increment_usage(username: str):
    """Increments the usage count for a user."""
    users = get_users()
    if username in users:
        # Ensure usage_count field exists
        if "usage_count" not in users[username]:
            users[username]["usage_count"] = 0
        users[username]["usage_count"] += 1
        save_users(users)
    else:
        # Create user entry if it doesn't exist
        users[username] = {
            "password": "",
            "name": username,
            "email": "",
            "usage_count": 1,
            "ip_address": ""
        }
        save_users(users)