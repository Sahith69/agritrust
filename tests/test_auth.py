"""
tests/test_auth.py - Unit tests for User Authentication & Unique Username Enforcement
"""

import os
import unittest
import tempfile
import auth


class TestAuth(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_auth.db")
        auth.init_user_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_registration_and_login_success(self):
        ok, msg = auth.register_user("farmer_ravi", "mypassword123", role="farmer", db_path=self.db_path)
        self.assertTrue(ok)
        self.assertIn("Account created", msg)

        # Verify correct credentials
        valid, user_data, msg = auth.verify_user("farmer_ravi", "mypassword123", db_path=self.db_path)
        self.assertTrue(valid)
        self.assertIsNotNone(user_data)
        self.assertEqual(user_data["username"], "farmer_ravi")
        self.assertEqual(user_data["role"], "farmer")

    def test_duplicate_username_prevented(self):
        ok1, _ = auth.register_user("farmer_anita", "secret123", db_path=self.db_path)
        self.assertTrue(ok1)

        # Attempt exact duplicate
        ok2, msg2 = auth.register_user("farmer_anita", "different_pass", db_path=self.db_path)
        self.assertFalse(ok2)
        self.assertIn("already taken", msg2)

        # Attempt case-insensitive duplicate ("Farmer_Anita")
        ok3, msg3 = auth.register_user("Farmer_Anita", "pass456", db_path=self.db_path)
        self.assertFalse(ok3)
        self.assertIn("already taken", msg3)

    def test_wrong_password_fails(self):
        auth.register_user("testuser", "correctpass", db_path=self.db_path)
        valid, user_data, msg = auth.verify_user("testuser", "wrongpass", db_path=self.db_path)
        self.assertFalse(valid)
        self.assertIsNone(user_data)
        self.assertIn("Incorrect password", msg)

    def test_nonexistent_user_fails(self):
        valid, user_data, msg = auth.verify_user("nonexistent", "pass123", db_path=self.db_path)
        self.assertFalse(valid)
        self.assertIsNone(user_data)
        self.assertIn("Username not found", msg)

    def test_empty_or_short_credentials(self):
        ok, msg = auth.register_user("", "pass123", db_path=self.db_path)
        self.assertFalse(ok)

        ok, msg = auth.register_user("ab", "pass123", db_path=self.db_path)
        self.assertFalse(ok)

        ok, msg = auth.register_user("validuser", "12", db_path=self.db_path)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
