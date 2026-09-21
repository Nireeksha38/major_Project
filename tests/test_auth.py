import pytest
from models.user import User


def test_password_hashing():
    user = User(full_name="Test Officer", email="officer@safety.gov")
    user.set_password("SecurePass@123")

    assert user.password != "SecurePass@123"
    assert user.check_password("SecurePass@123") is True
    assert user.check_password("WrongPassword") is False


def test_user_representation():
    user = User(full_name="Demo User", email="demo@example.com")
    assert "demo@example.com" in repr(user)
