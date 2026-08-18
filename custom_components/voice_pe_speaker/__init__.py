# pyright: reportMissingImports=false
"""Voice PE Speaker integration."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, Platform
from homeassistant.core import HomeAssistant

from .coordinator import VoicePESpeakerClient

PLATFORMS = [Platform.SENSOR]


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry[VoicePESpeakerClient]
) -> bool:
    """Set up Voice PE Speaker from a config entry."""
    client = VoicePESpeakerClient(hass, entry.data[CONF_HOST], entry.data[CONF_PORT])
    entry.runtime_data = client
    client.start()
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: ConfigEntry[VoicePESpeakerClient]
) -> bool:
    """Unload a Voice PE Speaker config entry."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    await entry.runtime_data.stop()
    return True
