# pyright: reportMissingImports=false
"""Sensor platform for Voice PE Speaker."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.components.sensor import SensorEntity

from .coordinator import SpeakerDetection, VoicePESpeakerClient


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry[VoicePESpeakerClient],
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Voice PE Speaker sensor."""
    async_add_entities([LastSpeakerSensor(entry)])


class LastSpeakerSensor(SensorEntity, RestoreEntity):
    """Represent the last verified speaker."""

    _attr_has_entity_name = True
    _attr_name = "Last Speaker"
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry[VoicePESpeakerClient]) -> None:
        self._client = entry.runtime_data
        self._attr_unique_id = f"{entry.entry_id}_last_speaker"
        self._restored_value: str | None = None
        self._restored_attributes: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        """Return whether the event stream is connected."""
        return self._client.connected

    @property
    def native_value(self) -> str | None:
        """Return the last verified speaker."""
        detection = self._client.detection
        if detection is not None:
            return detection.speaker or "unknown"
        return self._restored_value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return detection metadata."""
        detection = self._client.detection
        if detection is None:
            return self._restored_attributes
        return self._detection_attributes(detection)

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to push updates."""
        await super().async_added_to_hass()
        restored = await self.async_get_last_state()
        if restored is not None:
            self._restored_value = restored.state
            self._restored_attributes = dict(restored.attributes)
        self.async_on_remove(self._client.add_listener(self._handle_update))

    @callback
    def _handle_update(self) -> None:
        self.async_write_ha_state()

    @staticmethod
    def _detection_attributes(detection: SpeakerDetection) -> dict[str, Any]:
        return {
            "keyword": detection.keyword,
            "score": detection.score,
            "detected_at": detection.detected_at,
        }
