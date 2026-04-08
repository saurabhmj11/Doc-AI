"""
Authentication Router

Endpoints for user login, token exchange, and session management.
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from api.schemas import Token
from core.auth import (
    create_access_token, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    FAKE_USERS_DB,
    verify_password,
    get_current_active_user
)

router = APIRouter()


def authenticate_user_internal(username: str, password: str):
    user = FAKE_USERS_DB.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, retrieve a JWT token.
    """
    user = authenticate_user_internal(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me")
async def read_users_me(current_user: dict = Depends(get_current_active_user)):
    """
    Get current logged-in user information.
    """
    return current_user
