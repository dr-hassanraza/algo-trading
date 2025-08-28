
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
    
    with open(USERS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_users(users: dict):
    """Saves the users dictionary to the users.json file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def add_user(username: str, password: str) -> bool:
    """Adds a new user to the users.json file."""
    users = get_users()
    
    if username in users:
        return False  # User already exists
    
    users[username] = {
        "password": hash_password(password)
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
