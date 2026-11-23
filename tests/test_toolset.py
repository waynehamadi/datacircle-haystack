"""Offline tests: the tools against a local stand-in for the API on 127.0.0.1, by their own invoke, as Haystack
serializes them, and as an Agent runs them on a chat generator of the test's own."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import pytest
from haystack import component
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, Toolset
from haystack.tools.errors import ToolInvocationError

from haystack_integrations.tools.datacircle import (
    TOOL_NAMES,
    DatacircleError,
    DatacircleToolset,
    datacircle_tool,
    datacircle_tools,
)
from haystack_integrations.tools.datacircle import _api

PLAN = {"api": "Employee email", "rank": 1, "provider": "Hunter", "plan": "Starter", "score": 80.1, "perThousand": 4.9,
        "page": "https://datacircle.dev/x", "terms": "long", "aggregators": ["a"]}
SEEN = []
BUSY = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        SEEN.append((url.path, query, self.headers.get("User-Agent")))
        headers = {}
        if url.path == "/v1/plans" and query.get("query") == "busy" and not BUSY:
            BUSY.append(1)
            status, body, headers = 429, {"error": "slow down"}, {"Retry-After": "1"}
        elif url.path == "/v1/apis":
            status, body = 200, [{"api": "Employee email", "slug": "email-finder", "plans": 9, "providers": 3, "extra": 1},
                                 {"api": "Employee profile", "slug": "linkedin-profile", "plans": 4, "providers": 2}]
        elif url.path == "/v1/apis/email-finder/best":
            status, body = 200, {"api": "Employee email", "slug": "email-finder", "total": 1, "plans": [PLAN]}
        elif url.path.startswith("/v1/apis/"):
            status, body = 404, {"error": "no API has the slug nope", "slugs": ["email-finder", "linkedin-profile"]}
        elif url.path == "/v1/plans" and query.get("query") == "down":
            status, body = 503, {"error": "down for a minute"}
        elif url.path == "/v1/plans":
            status, body = 200, {"total": 1, "offset": 0, "plans": [PLAN]}
        elif url.path == "/v1/providers/bright-data":
            status, body = 200, {"provider": "Bright Data", "slug": "bright-data", "website": "https://brightdata.com",
                                 "apis": [{"api": "Employee email", "slug": "email-finder", "rank": 7, "sheetPage": "s",
                                           "plans": [dict(PLAN, rank=9, plan="B"), dict(PLAN, rank=7, plan="A")]}]}
        else:
            status, body = 404, {"error": "no provider has this slug"}
        data = json.dumps(body).encode()
        self.send_response(status)
        for key, value in {"Content-Type": "application/json", "Content-Length": str(len(data)), **headers}.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module", autouse=True)
def stand_in():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    before = _api.BASE_URL
    _api.BASE_URL = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield
    _api.BASE_URL = before
    server.shutdown()


def tool(name: str) -> Tool:
    return datacircle_tool(name)


def test_the_tools():
    toolset = DatacircleToolset()
    assert isinstance(toolset, Toolset) and [t.name for t in toolset] == TOOL_NAMES and len(toolset) == 4
    assert "datacircle_best_plans" in toolset and toolset[1].name == "datacircle_best_plans"
    for each in toolset:
        assert isinstance(each, Tool) and "https://datacircle.dev" in each.description
        assert each.tool_spec["parameters"]["type"] == "object"
    assert [t.name for t in DatacircleToolset(tool_names=["datacircle_provider", "datacircle_list_apis"])] == \
        ["datacircle_provider", "datacircle_list_apis"]
    assert [t.name for t in datacircle_tools()] == TOOL_NAMES
    assert list(DatacircleToolset(tool_names=[])) == []
    with pytest.raises(ValueError, match="its tools are datacircle_list_apis"):
        DatacircleToolset(tool_names=["datacircle_nope"])
    with pytest.raises(TypeError):
        DatacircleToolset(tool_names="datacircle_provider")
    with pytest.raises(ValueError, match="Duplicate"):
        DatacircleToolset(tool_names=["datacircle_provider", "datacircle_provider"])
    search = tool("datacircle_search_plans").parameters
    assert search["properties"]["sort"]["enum"] == ["score", "provider", "price"]
    assert search["properties"]["limit"] == {"type": "integer", "minimum": 1, "maximum": 50,
                                             "description": "How many plans to give, 1 to 50. 10 when left out."}
    assert tool("datacircle_best_plans").parameters["required"] == ["slug"]
    assert tool("datacircle_provider").parameters["required"] == ["slug"]


def test_each_tool_by_its_invoke():
    SEEN.clear()
    listed = tool("datacircle_list_apis").invoke(query="profile")
    assert listed == {"total": 1, "apis": [{"api": "Employee profile", "slug": "linkedin-profile", "plans": 4, "providers": 2}]}
    assert tool("datacircle_list_apis").invoke(limit=1)["apis"][0]["slug"] == "email-finder"
    best = tool("datacircle_best_plans").invoke(slug="Email Finder", limit=3)
    assert best["plans"] == [{key: value for key, value in PLAN.items() if key not in ("terms", "aggregators")}]
    found = tool("datacircle_search_plans").invoke(query="hunter", sort="price", limit=2)
    assert found["total"] == 1 and "terms" not in found["plans"][0]
    seller = tool("datacircle_provider").invoke(slug="Bright Data")
    assert seller["apis"] == [{"api": "Employee email", "slug": "email-finder", "rank": 7, "plans": 2,
                               "best": {"plan": "A", "score": 80.1, "perThousand": 4.9, "page": "https://datacircle.dev/x"}}]
    assert ("/v1/apis/email-finder/best", {"limit": "3"}, _api.USER_AGENT) in SEEN
    assert ("/v1/plans", {"query": "hunter", "sort": "price", "limit": "2"}, _api.USER_AGENT) in SEEN
    assert _api.USER_AGENT.startswith("datacircle-haystack/")
    if hasattr(Tool, "invoke_async"):
        assert asyncio.run(tool("datacircle_best_plans").invoke_async(slug="email-finder"))["total"] == 1


def refusal(name: str, **arguments: Any) -> str:
    with pytest.raises(ToolInvocationError) as raised:
        tool(name).invoke(**arguments)
    assert isinstance(raised.value.__cause__, DatacircleError)
    return str(raised.value)


def test_refusals_in_the_apis_words():
    words = refusal("datacircle_best_plans", slug="nope")
    assert "answered 404" in words and "email-finder, linkedin-profile" in words
    assert "no provider has this slug" in refusal("datacircle_provider", slug="nobody")
    assert "answered 503" in refusal("datacircle_search_plans", query="down")
    assert "Give a provider's slug" in refusal("datacircle_provider", slug="  ")
    with pytest.raises(DatacircleError) as raised:
        _api.get("/plans", {"query": "down"})
    assert raised.value.status == 503 and raised.value.retryable


def test_arguments_out_of_range_never_reach_the_api():
    SEEN.clear()
    assert "sort must be score, provider or price, not 'cheapest'" in refusal("datacircle_search_plans", sort="cheapest")
    assert "limit must be from 1 to 50, not 80" in refusal("datacircle_best_plans", slug="email-finder", limit=80)
    assert "offset must be a whole number" in refusal("datacircle_search_plans", offset=1.5)
    assert SEEN == []
    assert tool("datacircle_best_plans").invoke(slug="email-finder", limit="2")["total"] == 1  # a number sent as text


def test_waits_once_on_429():
    BUSY.clear()
    assert tool("datacircle_search_plans").invoke(query="busy")["total"] == 1


def test_the_model_reads_json():
    handler = tool("datacircle_best_plans").outputs_to_string["handler"]
    assert json.loads(handler({"slug": "é"})) == {"slug": "é"} and "é" in handler({"slug": "é"})


def test_serialized_as_its_names():
    data = DatacircleToolset(tool_names=["datacircle_best_plans"]).to_dict()
    assert data == {"type": "haystack_integrations.tools.datacircle.toolset.DatacircleToolset",
                    "data": {"tool_names": ["datacircle_best_plans"]}}
    back = DatacircleToolset.from_dict(json.loads(json.dumps(data)))
    assert [t.name for t in back] == ["datacircle_best_plans"]
    assert [t.name for t in DatacircleToolset.from_dict(DatacircleToolset().to_dict())] == TOOL_NAMES
    one = Tool.from_dict(json.loads(json.dumps(tool("datacircle_provider").to_dict())))
    assert one.invoke(slug="bright-data")["website"] == "https://brightdata.com"
    assert one.outputs_to_string["handler"]({"a": 1}) == '{"a": 1}'


@component
class Scripted:
    """A chat generator of the test's own: a call of each tool, two the API refuses, then the answer."""

    def __init__(self):
        self.seen: List[List[ChatMessage]] = []

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], tools: Optional[Any] = None, **kwargs: Any) -> Dict[str, Any]:
        self.seen.append(list(messages))
        turn = sum(1 for m in messages if m.is_from("assistant"))
        calls = [
            [ToolCall(tool_name="datacircle_list_apis", arguments={"query": "email"})],
            [ToolCall(tool_name="datacircle_best_plans", arguments={"slug": "email-finder", "limit": 1}),
             ToolCall(tool_name="datacircle_search_plans", arguments={"query": "hunter", "sort": "price"})],
            [ToolCall(tool_name="datacircle_provider", arguments={"slug": "Bright Data"}),
             ToolCall(tool_name="datacircle_best_plans", arguments={"slug": "nope"})],
        ]
        if turn < len(calls):
            return {"replies": [ChatMessage.from_assistant(tool_calls=calls[turn])]}
        return {"replies": [ChatMessage.from_assistant("done")]}


def test_an_agent_runs_the_four():
    agents = pytest.importorskip("haystack.components.agents")
    model = Scripted()
    agent = agents.Agent(chat_generator=model, tools=DatacircleToolset())
    if hasattr(agent, "warm_up"):
        agent.warm_up()
    result = agent.run(messages=[ChatMessage.from_user("Who is best for email finding?")])
    results = [m.tool_call_result for m in result["messages"] if m.tool_call_result is not None]
    assert [r.origin.tool_name for r in results] == ["datacircle_list_apis", "datacircle_best_plans",
                                                     "datacircle_search_plans", "datacircle_provider", "datacircle_best_plans"]
    assert json.loads(results[0].result)["apis"][0]["slug"] == "email-finder"
    assert json.loads(results[3].result)["website"] == "https://brightdata.com"
    assert results[4].error and "email-finder, linkedin-profile" in results[4].result
    assert result["messages"][-1].text == "done"
