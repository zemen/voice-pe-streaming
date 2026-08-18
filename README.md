# Voice PE Streaming

Streaming wake word + **speaker verification** for the **Home Assistant Voice PE**, and LLM
context so your assistant knows **who is speaking and where**.

- **Add-on**: LiveKit wake word + SpeechBrain ECAPA speaker verification + mid-sentence
  false-positive rejection over the Wyoming protocol, with a web UI for speaker enrollment and
  pipeline debugging.
- **HACS integration**: `sensor.last_speaker` — who last triggered the wake word, pushed live
  from the add-on. No tokens, no polling.
- **Firmware**: generated ESPHome config extending the official Voice PE package for
  streaming-only wake word.

## Installation

### 1. Add-on

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fzemen%2Fvoice-pe-streaming)

Click the badge (or add `https://github.com/zemen/voice-pe-streaming` under **Settings →
Add-ons → Add-on store → ⋮ → Repositories**), then install **Wyoming LiveKit + SpeechBrain**
and start it. The bundled `sputnik` wake word model is seeded automatically on first start.

Then connect it to Home Assistant:

[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=wyoming)

Add the **Wyoming Protocol** integration — Host: `localhost`, Port: `10400`. Pick the wake word
in **Settings → Voice assistants → your assistant → Wake word**.

Enroll speakers via the add-on's **Wake Word** panel in the HA sidebar (Ingress web UI).

### 2. Firmware

Flash `voice_pe_streaming/voice-pe-streaming.yaml` to your Voice PE with ESPHome. It extends
the official firmware; wake word detection moves to the add-on via your Assist pipeline.

Assign each Voice PE device to an **Area** in HA — the Assist LLM prompt automatically gains
"You are in area X", which answers the *where*.

### 3. HACS integration (the *who*)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=zemen&repository=voice-pe-streaming&category=integration)

Click the badge (or add this repo as a HACS custom repository, category *Integration*), install
**Voice PE Speaker**, then add the integration: **Settings → Devices & Services → Add** — Host:
your HA hostname, Port: `10401`. You get `sensor.last_speaker` with `keyword`, `score`, and
`detected_at` attributes, updated the moment a verified speaker wakes the device.

### 4. Tell the LLM who's speaking

Add this to your conversation agent's prompt template (Settings → Voice assistants → your
pipeline → conversation agent → Instructions):

```jinja
{% set last = states.sensor.last_speaker %}
{% if last is not none and last.state not in ('unknown', 'unavailable')
   and (now() - last.last_changed).total_seconds() < 30 %}
The person speaking to you right now is {{ last.state }}.
{% endif %}
```

The 30-second freshness window prevents stale attribution on typed chats or delayed queries.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

The full suite (800+ tests) runs without hardware and without the LiveKit/SpeechBrain SDKs
(mocked). Firmware is generated, not hand-edited: see `voice_pe_streaming/features.py` and
`scripts/generate_firmware.py`. The ESPHome parser/validator/simulator lives in
`esphome_yaml_tools/`.

## License

MIT
