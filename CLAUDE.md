# Aurora Audio Router — Project Context

## Repo
https://github.com/Rsmith-Partech/Aurora-routing-app

**Current version:** v1.3.0
**Platform:** Raspberry Pi (user: `par`, install dir: `~/audio-router/`)

---

## What this app does

Bidirectional USB audio cross-router for testing the LAi prototype audio interface.

- **Interface A** = WMT headset (mic + speaker)
- **Interface B** = LAi (prototype audio interface under active development/testing)
- Routes: Input A → Output B and Input B → Output A simultaneously

---

## Tech stack

| Component | Detail |
|---|---|
| GUI | Python 3.12 + Tkinter |
| Audio engine | `sounddevice` (PortAudio), `numpy` |
| Web UI | Flask, served on port 5000 |
| Install | `bash install.sh` — installs deps + desktop shortcut |

---

## Architecture

### Thread safety model
- **`RtState`** — lock-protected gains/mutes, read directly by audio callbacks (no Tkinter access)
- **`WebStatus`** — lock-protected struct Flask reads for running state, selected devices, sample rate, blocksize
- **`EventLog`** — lock-protected log buffer, callbacks posted to Tkinter via `after(0, ...)`
- **`web_cmd_q`** — `queue.Queue` for Flask → Tkinter commands (`start`, `stop`, `sync_rt`, `sync_ws`), drained every 100ms via `after()`

### Audio routing
- Each direction (`A→B`, `B→A`) is an independent `AudioRoute` instance
- Each `AudioRoute` owns an `InputStream` + `OutputStream` + `Queue(maxsize=32)`
- Drop-oldest queue overflow strategy keeps latency bounded
- Channel adaptation handles all mono/stereo conversions

### Web server
- Flask runs as a daemon thread (`use_reloader=False`, `werkzeug` logging suppressed)
- Full REST API: `/api/state`, `/api/devices`, `/api/start`, `/api/stop`, `/api/log`, `/api/log/download`, `/api/log/clear`
- Web UI is a single inline HTML string — no static files needed
- Pi's IP shown in Tkinter window and logged at startup

---

## Current features (v1.3.0)

- Per-channel gain sliders (default 25%) and mute controls
- Mute All master control (syncs bidirectionally with individual mutes)
- Automatic device hotplug detection + safety stop if selected device disappears
- Configurable sample rate (default 16 kHz) and blocksize (default 256)
- Browser UI at `http://<pi-ip>:5000` — full control parity with desktop
- Changes in browser reflect on desktop UI within ~100ms and vice versa

### Diagnostic logging (added for LAi dropout investigation)
- **Silence detection:** warns after 3s of near-zero RMS (`< 0.001`) on either input — catches stream-alive-but-silent dropout
- **xrun counting:** logs input overflow and output underflow with cumulative totals, throttled to once per 5s
- **Queue starvation:** warns after 10 consecutive empty output blocks (input feed stalled)
- **Stop summary:** logs per-session xrun totals on every stop
- **Device events:** connect/disconnect logged with full device string
- Download Log available from both desktop and browser

---

## Active problem under investigation

**Intermittent LAi audio dropout** — both inbound and outbound audio go silent simultaneously while the stream appears to still be running.

| Symptom | Implication |
|---|---|
| Stop → Start fixes it ~50–75% of the time | Stream state is recoverable without hardware reset |
| Power cycling the LAi fixes it otherwise | LAi firmware/USB audio stack freezes |
| No disconnect event fired | sounddevice doesn't detect the failure |
| Silence on both directions simultaneously | Not a one-way xrun cascade — likely device-side freeze |

**Hypothesis:** LAi USB audio firmware hangs, stops sending/accepting audio, but doesn't drop the USB device — so the OS and sounddevice see the stream as healthy. The diagnostic logging added in v1.2.0/v1.3.0 should help correlate silence events, xrun counts, and queue starvation timestamps when a dropout occurs.

**Next steps to consider:**
- Collect logs during a dropout incident and review silence/xrun timing
- Consider an auto-recovery watchdog: if silence detected on B→A input for >N seconds while not muted, automatically stop and restart routing
- Investigate whether increasing blocksize reduces dropout frequency

---

## User environment

- **Dev machine:** Windows 11 PC at `C:\Claude\`, PowerShell
- **Target:** Raspberry Pi on local network
- **GitHub:** `Rsmith-Partech`
- **Network:** TP-Link Deco mesh

---

## Workflow

```bash
# On the Pi — update and reinstall
git pull && bash install.sh

# Run directly
python3 ~/audio-router/audio_router.py

# Web UI
http://<pi-ip>:5000
```

**Always `git pull` and read `audio_router.py` before making changes — the repo is the source of truth.**
