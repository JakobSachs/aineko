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

## Development

```bash
# Run tests
uv run pytest

# Run locally (without container)
uv run aineko
```
