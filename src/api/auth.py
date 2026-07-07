"""
Sprint 9 — Authentication & Role-Based Access Control (SLO #12)

Roles (defined in Auth0 dashboard):
  l1-agent → submit tickets, view own history
  manager  → view all tickets, escalation logs, suspended accounts
  admin    → full access including RBAC management

Flow:
  React login → Auth0 issues JWT with roles claim
  React sends JWT in Authorization: Bearer <token> header
  FastAPI verifies signature against Auth0 JWKS → extracts roles → allow/deny
"""

import os
from functools import lru_cache

import httpx
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

load_dotenv()

_AUTH0_DOMAIN   = os.environ["AUTH0_DOMAIN"]
_API_AUDIENCE   = os.environ["AUTH0_API_AUDIENCE"]
_ROLES_CLAIM    = "https://support-resolution-api/roles"
_ALGORITHMS     = ["RS256"]

_bearer = HTTPBearer()


@lru_cache(maxsize=1)
def _get_jwks() -> dict:
    """Fetch Auth0 public keys — cached for the lifetime of the process."""
    url = f"https://{_AUTH0_DOMAIN}/.well-known/jwks.json"
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _verify_token(token: str) -> dict:
    jwks = _get_jwks()
    try:
        header = jwt.get_unverified_header(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token header")

    rsa_key = next(
        (
            {"kty": k["kty"], "kid": k["kid"], "use": k["use"], "n": k["n"], "e": k["e"]}
            for k in jwks["keys"]
            if k["kid"] == header.get("kid")
        ),
        None,
    )
    if not rsa_key:
        raise HTTPException(status_code=401, detail="No matching signing key found")

    try:
        return jwt.decode(
            token,
            rsa_key,
            algorithms=_ALGORITHMS,
            audience=_API_AUDIENCE,
            issuer=f"https://{_AUTH0_DOMAIN}/",
        )
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> dict:
    """FastAPI dependency — returns decoded JWT payload for any authenticated user."""
    return _verify_token(credentials.credentials)


def require_role(*roles: str):
    """
    FastAPI dependency factory — restricts endpoint to users with at least one of the given roles.

    Usage:
        @router.get("/admin-only")
        def admin_endpoint(user: dict = Depends(require_role("admin"))):
            ...

        @router.get("/managers-and-admins")
        def manager_endpoint(user: dict = Depends(require_role("manager", "admin"))):
            ...
    """
    def dependency(user: dict = Depends(get_current_user)) -> dict:
        user_roles = user.get(_ROLES_CLAIM, [])
        if not any(r in user_roles for r in roles):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required: {' or '.join(roles)}",
            )
        return user
    return dependency
