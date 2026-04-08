import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_login_success():
    response = client.post(
        "/api/auth/login",
        data={"username": "admin", "password": "admin"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_fail():
    response = client.post(
        "/api/auth/login",
        data={"username": "admin", "password": "wrongpassword"}
    )
    assert response.status_code == 401

def test_protected_route_no_token():
    response = client.post("/api/ask", json={"question": "test"})
    assert response.status_code == 401

def test_protected_route_with_token():
    # 1. Login
    login_resp = client.post(
        "/api/auth/login",
        data={"username": "admin", "password": "admin"}
    )
    token = login_resp.json()["access_token"]
    
    # 2. Access protected route
    # Note: This might still fail with 400/404 if no docs exist, but it shouldn't be 401
    response = client.post(
        "/api/ask",
        json={"question": "What is the rate?", "document_id": "nonexistent"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code != 401
