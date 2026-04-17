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

The app DB is SQLite at `/data/aineko.db` (mounted from `./data/` on the host). On container start, `alembic upgrade head` runs first, then the app launches.

> **Known issue:** `alembic.ini` currently hardcodes a relative URL (`sqlite+aiosqlite:///aineko.db`), so inside the container alembic migrates `/app/aineko.db` — a disposable file — not the real `/data/aineko.db`. Schema changes to existing prod tables won't apply until `alembic/env.py` is wired to use `Settings().db_url`. New tables still appear in prod because `db.create_tables()` runs on app startup.

## Development

```bash
# Run tests
uv run pytest -n4

# Run locally (without container)
uv run aineko
```
