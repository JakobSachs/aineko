"""Tests for config parsing."""

from aineko.config import MatrixSettings, KimiSettings, Settings


def test_matrix_room_list():
    s = MatrixSettings(rooms="!a:x,!b:x, !c:x ")
    assert s.room_list == ["!a:x", "!b:x", "!c:x"]


def test_matrix_room_list_empty():
    s = MatrixSettings(rooms="")
    assert s.room_list == []


def test_matrix_room_list_single():
    s = MatrixSettings(rooms="!room:test")
    assert s.room_list == ["!room:test"]


def test_kimi_defaults():
    s = KimiSettings()
    assert s.model == "kimi-latest"
    assert s.base_url == "https://api.moonshot.cn/v1"
    assert s.max_context_tokens == 128_000


def test_settings_db_url_default():
    s = Settings()
    assert "aineko.db" in s.db_url


def test_settings_db_url_override():
    s = Settings(database_url="sqlite:///custom.db")
    assert s.db_url == "sqlite:///custom.db"


def test_settings_skills_dir():
    s = Settings()
    assert str(s.skills_dir).endswith("skills")
