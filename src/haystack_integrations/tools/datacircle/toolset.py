"""Haystack tools on Datacircle's open API, https://api.datacircle.dev/v1: no key, no account.

Each tool makes one HTTPS GET and gives the model the API's answer as JSON text (Tool.invoke gives it as a dict). A
refusal (an unknown slug, an argument out of range, the API out of reach) raises a DatacircleError in the API's words:
the Agent, and ToolInvoker with raise_on_failure=False, send them to the model as the call's error, and the model can
call again. Haystack checks no argument against a tool's schema, so each tool checks its own before any call.
"""

import json
import urllib.parse
from typing import Any, Dict, List, Optional

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool, Toolset

from haystack_integrations.tools.datacircle import _api


def _best(plans: Any) -> Optional[Dict[str, Any]]:
    """A provider's best plan on an API: its lowest rank, a plan with no score having none."""
    ranked = sorted(plans, key=lambda plan: plan.get("rank") if isinstance(plan.get("rank"), (int, float)) else float("inf"))
    return _api.brief(ranked[0], ["plan", "score", "perThousand", "page"]) if ranked else None


def list_apis(query: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    count = _api.whole("limit", limit, 1, 200) or 20
    apis = _api.get("/apis")
    words = [word for word in str(query or "").lower().replace("-", " ").split() if word]
    held = []
    for index, api in enumerate(apis):
        name = f"{api.get('api', '')} {str(api.get('slug', '')).replace('-', ' ')}".lower()
        matched = sum(word in name for word in words)
        if matched or not words:
            held.append((-matched, index, api))
    found = [_api.brief(api, _api.API_KEYS) for _, _, api in sorted(held, key=lambda h: h[:2])]
    return {"total": len(found), "apis": found[:count]}


def best_plans(slug: str, limit: Optional[int] = None) -> Dict[str, Any]:
    made = _api.slug(slug)
    if not made:
        raise _api.DatacircleError("Give an API's slug, like 'email-finder': datacircle_list_apis gives every slug")
    answer = _api.get(f"/apis/{urllib.parse.quote(made)}/best", {"limit": _api.whole("limit", limit, 1, 50)})
    return {**answer, "plans": [_api.brief(plan) for plan in answer.get("plans") or []]}


def search_plans(api: Optional[str] = None, query: Optional[str] = None, sort: Optional[str] = None,
                 limit: Optional[int] = None, offset: Optional[int] = None) -> Dict[str, Any]:
    order = str(sort).strip().lower() if sort not in (None, "") else None
    if order is not None and order not in ("score", "provider", "price"):
        raise _api.DatacircleError(f"sort must be score, provider or price, not {sort!r}")
    params = {"api": api, "query": query, "sort": order, "limit": _api.whole("limit", limit, 1, 50),
              "offset": _api.whole("offset", offset, 0)}
    answer = _api.get("/plans", params)
    return {**answer, "plans": [_api.brief(plan) for plan in answer.get("plans") or []]}


def provider(slug: str) -> Dict[str, Any]:
    made = _api.slug(slug)
    if not made:
        raise _api.DatacircleError("Give a provider's slug or name, like 'bright-data' or 'Bright Data'")
    answer = _api.get(f"/providers/{urllib.parse.quote(made)}")
    apis = [{**_api.brief(entry, ["api", "slug", "rank", "bestPage"]), "plans": len(entry.get("plans") or []),
             "best": _best(entry.get("plans") or [])} for entry in answer.get("apis") or []]
    return {**answer, "apis": apis}


def as_text(result: Any) -> str:
    """What the model reads of a tool's answer: its JSON."""
    return json.dumps(result, ensure_ascii=False)


_PRICE = "perThousand is a plan's price of 1,000 items in dollars, pay the plan's whole price in its currency."
_LIMIT = {"type": "integer", "minimum": 1, "maximum": 50, "description": "How many plans to give, 1 to 50. 10 when left out."}

# Each tool: its description, its parameters' JSON schema and its function.
_TOOLS: Dict[str, Any] = {
    "datacircle_list_apis": (
        "Lists the GTM APIs on Datacircle's sheet (https://datacircle.dev), with each API's slug, how many plans and "
        "providers it has and its best plan. Call it first to find an API's slug. Gives total, the APIs found, and "
        "apis, up to limit of them.",
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "Words of the API's name or slug, like 'email' or 'linkedin "
                                                       "profile': the APIs that hold the most of them come first. The "
                                                       "sheet's order when left out."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200,
                      "description": "How many APIs to give, 1 to 200. 20 when left out: the sheet has about 160."},
        }},
        list_apis,
    ),
    "datacircle_best_plans": (
        "Gives an API's plans best first by their score out of 100 on Datacircle's benchmark (https://datacircle.dev), "
        "each with its rank, provider, plan, $ per 1,000 calls and its page: the best providers of an API. " + _PRICE,
        {"type": "object", "properties": {
            "slug": {"type": "string", "description": "The API's slug, like 'email-finder' or 'linkedin-profile', as "
                                                      "datacircle_list_apis gives it."},
            "limit": _LIMIT,
        }, "required": ["slug"]},
        best_plans,
    ),
    "datacircle_search_plans": (
        "Searches every plan on Datacircle's sheet (https://datacircle.dev), or one API's, by provider or plan name, "
        "sorted by score, provider or price. Gives total, offset and up to limit plans. " + _PRICE,
        {"type": "object", "properties": {
            "api": {"type": "string", "description": "Only this API's plans: its name or slug, like 'email-finder'. A "
                                                     "provider's name goes in query, never here. Every API when left out."},
            "query": {"type": "string", "description": "Words that must all appear in the provider's or plan's name, "
                                                       "like 'bright data'."},
            "sort": {"type": "string", "enum": ["score", "provider", "price"],
                     "description": "score (best first, the default), provider (A to Z) or price ($ per 1,000, lowest first)."},
            "limit": _LIMIT,
            "offset": {"type": "integer", "minimum": 0, "description": "How many found plans to skip first. 0 when left out."},
        }},
        search_plans,
    ),
    "datacircle_provider": (
        "Gives each API one provider sells and its rank on each on Datacircle's benchmark (https://datacircle.dev), "
        "with its best plan there.",
        {"type": "object", "properties": {
            "slug": {"type": "string", "description": "The provider's slug, like 'bright-data', or its name, like "
                                                      "'Bright Data': the name is made a slug."},
        }, "required": ["slug"]},
        provider,
    ),
}
TOOL_NAMES = list(_TOOLS)


def datacircle_tool(name: str) -> Tool:
    """One of the four tools, by its name: a plain Haystack Tool, serialized as any other."""
    if name not in _TOOLS:
        raise ValueError(f"Datacircle has no tool {name!r}: its tools are {', '.join(TOOL_NAMES)}")
    description, parameters, function = _TOOLS[name]
    return Tool(name=name, description=description, parameters=json.loads(json.dumps(parameters)), function=function,
                outputs_to_string={"handler": as_text})


class DatacircleToolset(Toolset):
    """The four tools, one for each call of the API, for Agent(chat_generator=..., tools=DatacircleToolset()), a
    ToolInvoker or a chat generator's tools. tool_names keeps some of them only, in their order here; all four when
    left out. Serialized as its tool_names alone, so a pipeline loaded later gets the tools of the version installed."""

    def __init__(self, tool_names: Optional[List[str]] = None):
        if isinstance(tool_names, str):
            raise TypeError("tool_names is a list of names, like ['datacircle_best_plans']")
        self.tool_names = list(tool_names) if tool_names is not None else None
        super().__init__(tools=[datacircle_tool(name) for name in (TOOL_NAMES if self.tool_names is None else self.tool_names)])

    def to_dict(self) -> Dict[str, Any]:
        return {"type": generate_qualified_class_name(type(self)), "data": {"tool_names": self.tool_names}}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatacircleToolset":
        return cls(tool_names=data.get("data", {}).get("tool_names"))


def datacircle_tools(tool_names: Optional[List[str]] = None) -> List[Tool]:
    """The same tools as a list, for a list of tools of one's own."""
    return list(DatacircleToolset(tool_names).tools)


__all__: List[str] = ["DatacircleToolset", "TOOL_NAMES", "as_text", "best_plans", "datacircle_tool", "datacircle_tools",
                      "list_apis", "provider", "search_plans"]
