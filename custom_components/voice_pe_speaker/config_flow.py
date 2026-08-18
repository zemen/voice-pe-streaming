# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportCallIssue=false
"""Config flow for Voice PE Speaker."""

from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import ClientError
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.config_entries import ConfigFlowResult
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.helpers import aiohttp_client

from .const import DEFAULT_HOST, DEFAULT_PORT, DOMAIN


class VoicePESpeakerConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Voice PE Speaker."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial user step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            host = user_input[CONF_HOST]
            port = user_input[CONF_PORT]
            try:
                async with asyncio.timeout(5):
                    session = aiohttp_client.async_get_clientsession(self.hass)
                    async with session.get(f"http://{host}:{port}/api/version") as response:
                        response.raise_for_status()
            except (ClientError, OSError, TimeoutError):
                errors["base"] = "cannot_connect"
            else:
                await self.async_set_unique_id(f"{host}:{port}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(title=f"{host}:{port}", data=user_input)

        schema = vol.Schema(
            {
                vol.Required(CONF_HOST, default=DEFAULT_HOST): str,
                vol.Required(CONF_PORT, default=DEFAULT_PORT): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=65535)
                ),
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)
