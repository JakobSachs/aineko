"""Configuration via pydantic-settings. Reads from env vars and /data/config.json."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MatrixSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MATRIX_")

    homeserver: str = "https://matrix.org"
    user_id: str = ""
    access_token: str = ""
    password: str = ""  # used for initial login if no access_token
    rooms: str = ""  # comma-separated room IDs

    @property
    def room_list(self) -> list[str]:
        return [r.strip() for r in self.rooms.split(",") if r.strip()]


class KimiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KIMI_")

    api_key: str = ""
    model: str = "kimi-k2.5"
    base_url: str = "https://api.moonshot.cn/v1"
    max_context_tokens: int = 262_144
    user_agent: str = ""  # required for some endpoints (e.g. Kimi coding API)
    thinking: bool = True  # enable reasoning mode (kimi-k2.5, kimi-k2-thinking)
    temperature: float = 1.0  # 1.0 for thinking models, 0.6 for non-thinking
    top_p: float = 0.95  # recommended for kimi-k2.5
    max_tokens: int = 32_000  # output token limit


class HeartbeatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HEARTBEAT_")

    enabled: bool = True
    every_minutes: int = 30
    active_hours_start: str = "08:00"
    active_hours_end: str = "23:00"
    room: str = ""  # defaults to first matrix room if empty


class CronSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRON_")

    enabled: bool = True
    max_concurrent_runs: int = 1


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AINEKO_")

    data_dir: Path = Path("/data")
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"
    database_url: str = ""  # default built from data_dir in property

    brave_api_key: str = ""  # Brave Search API key

    matrix: MatrixSettings = Field(default_factory=MatrixSettings)
    kimi: KimiSettings = Field(default_factory=KimiSettings)
    heartbeat: HeartbeatSettings = Field(default_factory=HeartbeatSettings)
    cron: CronSettings = Field(default_factory=CronSettings)

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite+aiosqlite:///{self.data_dir / 'aineko.db'}"

    @property
    def skills_dir(self) -> Path:
        return self.data_dir / "skills"

    @property
    def heartbeat_file(self) -> Path:
        return self.data_dir / "HEARTBEAT.md"
