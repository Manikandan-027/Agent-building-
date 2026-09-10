"""Authentication & principals.

API keys are stored ONLY as SHA-256 hashes. Two modes:
  - configured hashes (ARA_API_KEY_SHA256) for production
  - a plaintext dev key (ARA_DEV_API_KEY) hashed at verify time for dev/CI
Tenant + role come from the key record: users/api_keys tables (dev auto-seeds a
tenant admin), falling back to a deterministic dev principal in test mode.
"""
from __future__ import annotations

from ara.core.config import Settings
from ara.core.errors import AuthenticationError
from ara.core.ids import iso_now, new_id
from ara.db import UnitOfWork
from ara.policy.authz import Principal


def ensure_dev_principal(uow: UnitOfWork, settings: Settings) -> Principal:
    """Seed tenant/user rows for the dev key so foreign keys and scoping work."""
    tenant_id, user_id = "ten_dev", "user_dev_admin"
    if not uow.db.query_one("SELECT id FROM users WHERE id=?", (user_id,)):
        uow.db.execute(
            "INSERT INTO users (id, tenant_id, email, role, created_at) VALUES (?,?,?,?,?)",
            (user_id, tenant_id, "dev@ara.local", "admin", iso_now()))
    if not uow.db.query_one("SELECT id FROM api_keys WHERE key_hash=?", (settings.hash_api_key(settings.dev_api_key),)):
        uow.db.execute(
            "INSERT INTO api_keys (id, tenant_id, user_id, key_hash, label, created_at) VALUES (?,?,?,?,?,?)",
            (new_id("key"), tenant_id, user_id, settings.hash_api_key(settings.dev_api_key),
             "dev key", iso_now()))
    return Principal.for_role(user_id, tenant_id, "admin")


class Authenticator:
    def __init__(self, uow: UnitOfWork, settings: Settings):
        self.uow = uow
        self.settings = settings
        self._dev = ensure_dev_principal(uow, settings)
        self._configured = settings.accepted_api_key_hashes()

    def authenticate(self, api_key: str | None) -> Principal:
        if not api_key:
            raise AuthenticationError("missing X-API-Key header")
        key_hash = self.settings.hash_api_key(api_key)
        if self._configured and key_hash in self._configured:
            # configured production keys map to the tenant implied by their record
            row = self.uow.db.query_one("SELECT * FROM api_keys WHERE key_hash=? AND revoked=0", (key_hash,))
            if row:
                return Principal.for_role(row["user_id"], row["tenant_id"], self._role_of(row["user_id"]))
            raise AuthenticationError("unknown api key")
        if api_key == self.settings.dev_api_key and self.settings.env != "production":
            return self._dev
        row = self.uow.db.query_one("SELECT * FROM api_keys WHERE key_hash=? AND revoked=0", (key_hash,))
        if not row:
            raise AuthenticationError("invalid api key")
        return Principal.for_role(row["user_id"], row["tenant_id"], self._role_of(row["user_id"]))

    def _role_of(self, user_id: str) -> str:
        row = self.uow.db.query_one("SELECT role FROM users WHERE id=?", (user_id,))
        return (row or {}).get("role", "user")

    def create_key(self, *, tenant_id: str, user_id: str, role: str = "user",
                   label: str = "") -> tuple[str, str]:
        """Provision a new API key. Returns (api_key, key_id). The plaintext key is
        shown ONCE; only its hash is stored."""
        import secrets

        api_key = f"ara-{secrets.token_urlsafe(24)}"
        if role == "admin" and not self.uow.db.query_one("SELECT id FROM users WHERE id=?", (user_id,)):
            uow = self.uow
            uow.db.execute("INSERT INTO users (id, tenant_id, email, role, created_at) VALUES (?,?,?,?,?)",
                           (user_id, tenant_id, f"{user_id}@keys.local", role, iso_now()))
        kid = new_id("key")
        self.uow.db.execute(
            "INSERT INTO api_keys (id, tenant_id, user_id, key_hash, label, created_at) VALUES (?,?,?,?,?,?)",
            (kid, tenant_id, user_id, self.settings.hash_api_key(api_key), label, iso_now()))
        return api_key, kid
