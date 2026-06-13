"""Configuration via pydantic-settings. Reads from env vars and /data/config.json."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MatrixSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MATRIX_")

    homeserver: str = "https://matrix.org"
    user_id: str = ""
    access_token: str = ""
    password: str = ""  # used for initial login if no access_token
    room_id: str = ""  # the single matrix room this bot talks in
    owner: str = ""  # only this matrix user is allowed to talk to the bot


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: str = "litellm"
    api_key: str = ""
    model: str = "moonshot/kimi-k2.5"
    base_url: str = ""
    opencode_auth_path: Path = Path("/opencode-auth/auth.json")
    max_context_tokens: int = 32_000
    compaction_keep_recent: int = 4  # messages to keep when compacting
    user_agent: str = ""  # required for some endpoints
    thinking: bool = True  # provider-specific reasoning mode when supported
    temperature: float = 1.0  # 1.0 for thinking models, 0.6 for non-thinking
    top_p: float = 0.95
    max_tokens: int = 32_000  # output token limit
    memory_flush_enabled: bool = True  # run pre-compaction memory flush turn


class KimiSettings(LLMSettings):
    """Backward-compatible env prefix for existing deployments."""

    model_config = SettingsConfigDict(env_prefix="KIMI_")
    model: str = "kimi-k2.5"
    base_url: str = "https://api.moonshot.cn/v1"


class HeartbeatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HEARTBEAT_")

    enabled: bool = True
    every_minutes: int = 30
    active_hours_start: str = "08:00"
    active_hours_end: str = "23:00"


class CalDavSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CALDAV_")

    url: str = "https://caldav.icloud.com"
    username: str = ""  # Apple ID email
    password: str = ""  # App-specific password


class CronSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRON_")

    enabled: bool = True
    max_concurrent_runs: int = 1


class RssFeedConfig(BaseModel):
    url: str
    name: str


class RssSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RSS_")

    enabled: bool = False
    poll_interval: int = 300  # seconds


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AINEKO_")

    data_dir: Path = Path("/data")
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"
    database_url: str = ""  # default built from data_dir in property

    brave_api_key: str = ""  # Brave Search API key

    matrix: MatrixSettings = Field(default_factory=MatrixSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    kimi: KimiSettings = Field(default_factory=KimiSettings)
    heartbeat: HeartbeatSettings = Field(default_factory=HeartbeatSettings)
    caldav: CalDavSettings = Field(default_factory=CalDavSettings)
    cron: CronSettings = Field(default_factory=CronSettings)
    rss: RssSettings = Field(default_factory=RssSettings)

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return "postgresql+asyncpg://aineko:aineko@postgres:5432/aineko"

    @property
    def skills_dir(self) -> Path:
        return self.data_dir / "skills"

    @property
    def heartbeat_file(self) -> Path:
        return self.data_dir / "HEARTBEAT.md"

    @property
    def rss_feeds_file(self) -> Path:
        return self.data_dir / "rss_feeds.json"
