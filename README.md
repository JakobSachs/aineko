# aineko

Personal AI gateway: Matrix chat interface, Kimi AI brain, self-modifying skills.

## Running

Aineko runs as a podman container managed by a systemd user service.

```bash
# Rebuild and restart
./scripts/rebuild.sh

# Manual control
systemctl --user start aineko
systemctl --user stop aineko
systemctl --user restart aineko
systemctl --user status aineko

# Logs
podman logs -f aineko_aineko_1
```

The service unit lives at `~/.config/systemd/user/aineko.service` and is enabled on login (`WantedBy=default.target`).

### Restart policy

Restarts are owned by **systemd** (`Restart=on-failure` in the unit), not by podman. `docker-compose.yml` intentionally has no `restart:` policy — if both layers tried to restart the container, `podman-compose up` could exit after podman auto-restarted a crashed container, leaving systemd and the container out of sync (service shows inactive while the container keeps running). Keeping one source of truth avoids that.

### Database and migrations

The app DB is Postgres, run as a sidecar `postgres` service in `docker-compose.yml` (image `postgres:17-alpine`, data in the `postgres-data` volume, exposed on `127.0.0.1:5432`). The aineko container `depends_on` postgres being healthy before starting.

Connection URL resolution lives in `Settings.db_url` (`src/aineko/config.py`): it uses `AINEKO_DATABASE_URL` / `database_url` if set, otherwise defaults to `postgresql+asyncpg://aineko:aineko@postgres:5432/aineko`. Credentials and DB name come from `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` env vars (defaults: `aineko`/`aineko`/`aineko`).

On container start, `alembic upgrade head` runs first, then the app launches. `alembic/env.py` reads the URL from `AINEKO_DATABASE_URL` or `Settings().db_url`, so migrations hit the real DB (the old SQLite-on-/app quirk is gone).

## Development

```bash
# Run tests
uv run pytest -n4

# Run locally (without container)
uv run aineko
```
