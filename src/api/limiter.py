from slowapi import Limiter
from slowapi.util import get_remote_address

# Single limiter instance shared by main.py (registers with app) and routes.py (decorates endpoints).
# Keyed by client IP address — 10 requests/minute per IP (SLO #14 cost guard).
limiter = Limiter(key_func=get_remote_address)
