from functools import wraps

from flask import jsonify, request, session


def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            return jsonify({"error": "Authentication required."}), 401
        return func(*args, **kwargs)
    return wrapper


def wants_json() -> bool:
    return request.is_json or "application/json" in request.headers.get("Accept", "")
