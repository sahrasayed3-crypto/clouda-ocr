# Ubuntu/Debian deployment

1. Create a dedicated unprivileged user named `clouda` and place the project at `/opt/clouda`.
2. Run `deploy/linux/install.sh` as that user. It installs server-only Python
   dependencies; OCR dependencies belong on the worker.
3. Copy `clouda.env.example` to `clouda.env` and set only the required values.
4. As an administrator, copy `clouda.service` to `/etc/systemd/system/`.
5. Run `sudo systemctl daemon-reload && sudo systemctl enable --now clouda`.
6. Check with `systemctl status clouda` and `deploy/linux/health_check.sh`.

See `DISTRIBUTED_DEPLOYMENT.md` for Redis, the internal API, and Windows worker
configuration. Do not expose Redis or the internal API to the public internet.

Schedule `cleanup.sh` and the application Backup action according to the host retention policy.
The included `clouda-cleanup.timer` and `clouda-backup.timer` provide daily defaults; copy both
timer/service pairs to `/etc/systemd/system/`, then enable them with
`sudo systemctl enable --now clouda-cleanup.timer clouda-backup.timer`.
