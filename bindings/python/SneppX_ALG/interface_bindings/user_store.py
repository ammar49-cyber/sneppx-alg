"""User account store — registration, authentication, account management."""

import os
import sqlite3
import hashlib
import logging
import threading
import uuid
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_USER_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    tier            TEXT NOT NULL DEFAULT 'free',
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      REAL NOT NULL,
    last_login_at   REAL
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
"""


class UserStore:
    """SQLite-backed user account store with password authentication."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or os.environ.get(
            "SNEPPX_USER_DB_PATH",
            os.path.join(os.path.expanduser("~"), ".sneppx", "users.db"),
        )
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(_USER_SCHEMA)
        conn.commit()

    @staticmethod
    def _hash_password(password: str, salt: Optional[str] = None) -> str:
        if salt is None:
            salt = uuid.uuid4().hex[:16]
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return f"{salt}${h.hex()}"

    def _check_password(self, password: str, stored: str) -> bool:
        parts = stored.split("$")
        if len(parts) != 2:
            return False
        salt, _ = parts
        return self._hash_password(password, salt) == stored

    def create_user(self, email: str, password: str, tier: str = "free") -> Optional[Dict[str, Any]]:
        """Register a new user. Returns user info dict (without password_hash)."""
        conn = self._get_conn()
        user_id = uuid.uuid4().hex
        password_hash = self._hash_password(password)
        now = time.time()
        try:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, tier, is_active, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (user_id, email, password_hash, tier, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return None
        logger.info("Created user %s (%s)", user_id[:8], email)
        return {"id": user_id, "email": email, "tier": tier, "created_at": now}

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user by email and password. Returns user info or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, email, password_hash, tier, is_active, created_at, last_login_at "
            "FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if row is None:
            return None
        user = dict(row)
        if not user["is_active"]:
            return None
        if not self._check_password(password, user["password_hash"]):
            return None
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (time.time(), user["id"]))
        conn.commit()
        del user["password_hash"]
        return user

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID (without password_hash)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, email, tier, is_active, created_at, last_login_at "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, email, tier, is_active, created_at, last_login_at "
            "FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        return dict(row) if row else None

    def update_tier(self, user_id: str, tier: str) -> bool:
        conn = self._get_conn()
        cur = conn.execute("UPDATE users SET tier = ? WHERE id = ?", (tier, user_id))
        conn.commit()
        return cur.rowcount > 0


_GLOBAL_USER_STORE: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _GLOBAL_USER_STORE
    if _GLOBAL_USER_STORE is None:
        _GLOBAL_USER_STORE = UserStore()
    return _GLOBAL_USER_STORE


def set_user_store(store: UserStore):
    global _GLOBAL_USER_STORE
    _GLOBAL_USER_STORE = store


__all__ = ["UserStore", "get_user_store", "set_user_store"]
