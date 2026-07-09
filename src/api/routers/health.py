from fastapi import APIRouter, Depends

from src.api.auth import get_current_user

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    """Returns the current user's identity and roles — used by frontend for role-based UI."""
    return {
        "sub":   user.get("sub"),
        "roles": user.get("https://support-resolution-api/roles", []),
    }
