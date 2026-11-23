"""Haystack tools on Datacircle's open API (https://api.datacircle.dev/v1), no key:
Agent(chat_generator=..., tools=DatacircleToolset()), or any of the four in a list of tools."""

from haystack_integrations.tools.datacircle._api import DatacircleError
from haystack_integrations.tools.datacircle.toolset import (
    TOOL_NAMES,
    DatacircleToolset,
    datacircle_tool,
    datacircle_tools,
)

__version__ = "0.1.0"
__all__ = ["TOOL_NAMES", "DatacircleError", "DatacircleToolset", "datacircle_tool", "datacircle_tools"]
