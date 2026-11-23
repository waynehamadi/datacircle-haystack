# datacircle-haystack

Find the best GTM APIs. We benchmark all the GTM APIs and we put the results on our website.

Our website: https://datacircle.dev

This package gives [Haystack](https://github.com/deepset-ai/haystack) four tools that call Datacircle's public API, https://api.datacircle.dev/v1. You need no key and no account.

| Tool | Gives | Call |
| --- | --- | --- |
| `datacircle_list_apis` | Every GTM API on Datacircle, with the API's slug, number of plans, number of providers and best plan | `GET /apis` |
| `datacircle_best_plans` | One API's plans, sorted by score out of 100, best first. Each plan comes with a rank and a price per 1,000 calls | `GET /apis/{slug}/best` |
| `datacircle_search_plans` | Plans that match a provider or plan name, on all APIs or on one API. You can sort them by score, provider or price | `GET /plans` |
| `datacircle_provider` | The APIs a provider sells, and the provider's rank on each API | `GET /providers/{slug}` |

## Install

```bash
pip install datacircle-haystack
```

The package needs Haystack 2.15 or later (`haystack-ai`) and Python 3.10 or later.

## Give the tools to an agent

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.datacircle import DatacircleToolset

agent = Agent(chat_generator=OpenAIChatGenerator(), tools=DatacircleToolset())
agent.warm_up()
result = agent.run(messages=[ChatMessage.from_user("Which email finder API is best for under $5 per 1,000?")])
print(result["last_message"].text)
```

`OpenAIChatGenerator` reads your key from `OPENAI_API_KEY`. Any Haystack chat generator that calls tools works the same way.

To give the agent fewer tools, list their names: `DatacircleToolset(tool_names=["datacircle_best_plans"])`.

You can also use the toolset in an `Agent` inside a pipeline. On Haystack 2, you can pass the toolset to a chat generator's `tools` and to a `ToolInvoker`. Set `raise_on_failure=False` on the `ToolInvoker` so the model sees the error message. Save a pipeline and Haystack stores the toolset as a list of tool names. Update this package, and the saved pipeline gets the new tools.

## Call a tool yourself

Each tool returns our API's answer as a dict:

```python
from haystack_integrations.tools.datacircle import datacircle_tool

print(datacircle_tool("datacircle_best_plans").invoke(slug="email-finder", limit=3))
```

## Tool output for the model

The model gets each answer as JSON. Each plan comes with a rank, provider, plan, score, price per 1,000 (`perThousand`, in dollars) and a page on datacircle.dev. That page lists every field.

You can give a provider's name in place of a slug. The tool turns `Bright Data` into `bright-data`.

If the model passes an unknown slug, the tool raises a `DatacircleError` with the API's error message. For an unknown API slug, the message lists every valid slug. The Agent gives the message to the model, and the model can call the tool again. The tool checks each number from the model before the API call. If a number is out of range, the tool skips the API call. The model gets an error message.

## Network

Each tool call makes one HTTPS GET request to https://api.datacircle.dev/v1 and nothing else. The request holds only the tool's own arguments. The API takes up to 60 calls a minute from one IP address. When the API asks the tool to wait, the tool waits up to a minute. The tool waits once per call at most. All plans are in one file at https://datacircle.dev/datacircle.json.

## Links

- Datacircle: https://datacircle.dev
- The API: https://datacircle.dev/api
- The API's spec: https://api.datacircle.dev/v1/openapi.json
- What each key means: https://datacircle.dev/fields

MIT license.
