# zenai

Zig client for AI APIs, ported from the official Go SDKs ([go-genai](https://github.com/googleapis/go-genai), [openai-go](https://github.com/openai/openai-go), [anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go)). It covers:

- **LLMs**
  - [Google Gemini](https://ai.google.dev/gemini-api/docs), on the Developer API or [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs)
  - [OpenAI](https://platform.openai.com/docs/api-reference), plus any OpenAI-compatible endpoint through the same client:
    - Local: [Ollama](https://github.com/ollama/ollama/blob/main/docs/openai.md), [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) (`llama-server`), [LLMD](https://zml.ai/llmd/)
    - Hosted: [Hugging Face Inference](https://huggingface.co/docs/inference-providers/index), [Mistral](https://docs.mistral.ai/), [xAI](https://docs.x.ai/)
    - Gateways: [OpenRouter](https://openrouter.ai/docs), [Vercel AI Gateway](https://vercel.com/docs/ai-gateway)
    - Anything else, via `OPENAI_BASE_URL`
  - [Codex](#chatgpt-subscription-codex): OpenAI models billed to your ChatGPT plan instead of API credits
  - [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models)
- **Search** (`zenai.search`): [Tavily](https://docs.tavily.com/), [Brave Search](https://brave.com/search/api/), [Exa](https://exa.ai/docs/reference/search), [Keenable](https://docs.keenable.ai/), [Parallel](https://docs.parallel.ai/search/search-quickstart)
- **Judgement** (`zenai.typesafe`): [TypeSafe System One](https://docs.typesafe.ai), whose Jev model answers typed questions about a piece of state (a probability, a labelled choice or a score) instead of generating text

<img width="1024" height="1024" alt="Meditating panda with incense smoke" src="https://github.com/user-attachments/assets/b9c82960-05ec-4aa1-b171-092ee2126551" />

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Gemini](#gemini) · [OpenAI](#openai) · [Codex](#chatgpt-subscription-codex) · [Anthropic](#anthropic)
- [Search](#search)
- [TypeSafe](#typesafe-system-one)
- [Features](#features)

## Installation

```bash
zig fetch --save git+https://github.com/lightpanda-io/zenai
```

Then add the dependency in your `build.zig`:

```zig
const zenai = b.dependency("zenai", .{});
exe.root_module.addImport("zenai", zenai.module("zenai"));
```

Requires Zig >= 0.17.0. The examples below assume `allocator`, `io`, and `environ` are in scope; with Zig's `main` signature they come straight from `std.process.Init`:

```zig
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const environ = init.minimal.environ;
    // ...
}
```

## Quick start

`zenai.provider.Client` wraps every LLM backend behind one interface, so switching providers means changing the line that builds it:

```zig
const zenai = @import("zenai");

var gemini_client = zenai.gemini.Client.init(io, allocator, gemini_key, .{});
defer gemini_client.deinit();
const ai: zenai.provider.Client = .{ .gemini = &gemini_client };

var result = try ai.generateContent("gemini-2.5-flash", &.{
    .{ .role = .user, .content = "What is Zig?" },
}, .{});
defer result.deinit();

std.debug.print("{s}\n", .{result.text orelse ""});
```

The OpenAI-compatible backends all use the OpenAI client with a different `base_url` and tag:

| Tag | Default `base_url` | Key |
| --- | --- | --- |
| `.openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `.ollama` | `http://localhost:11434/v1` | none |
| `.llama_cpp` | `http://localhost:8080/v1` | none |
| `.llmd` | `http://localhost:8000/v1` | none |
| `.huggingface` | `https://router.huggingface.co/v1` | `HF_TOKEN` |
| `.mistral` | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` |
| `.xai` | `https://api.x.ai/v1` | `XAI_API_KEY` |
| `.openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `.vercel` | `https://ai-gateway.vercel.sh/v1` | `AI_GATEWAY_API_KEY` |
| `.openai_compatible` | `OPENAI_BASE_URL` | `OPENAI_API_KEY` |

```zig
var llama_client = zenai.llama_cpp.Client.init(io, allocator, "llama.cpp", .{
    .base_url = "http://localhost:8080/v1",
});
const ai: zenai.provider.Client = .{ .llama_cpp = &llama_client };
```

Gateway model ids are provider-prefixed (`"anthropic/claude-sonnet-5"`). For Hugging Face, pass `.base_url` to target a dedicated Inference Endpoint instead of the serverless router. `provider.Client.init` picks `.openai_compatible` on its own when `OPENAI_BASE_URL` is set, which covers vLLM, LiteLLM, Together, Groq and the like.

When you need something the abstraction doesn't expose, switch on the tag and use the provider client directly:

```zig
switch (ai) {
    .gemini => |g| {
        // Use Gemini-specific features like cached content, file uploads, etc.
        var cached = try g.createCachedContent("gemini-2.5-flash", .{ ... });
    },
    else => {},
}
```

## Gemini

Set your API key ([get one here](https://ai.google.dev/gemini-api/docs/api-key)):

```bash
export GOOGLE_API_KEY='your-api-key'
```

```zig
const zenai = @import("zenai");

const api_key = environ.getPosix("GOOGLE_API_KEY") orelse return error.MissingApiKey;
var client = zenai.gemini.Client.init(io, allocator, api_key, .{});
defer client.deinit();

var response = try client.generateContentFromText("gemini-2.5-flash", "What is Zig?", .{}, .{});
defer response.deinit();

std.debug.print("{s}\n", .{response.value.text() orelse ""});
```

<details>
<summary>Streaming</summary>

```zig
try client.generateContentStreamFromText(
    "gemini-2.5-flash",
    "Write a poem about the moon.",
    .{},
    .{},
    {},
    &struct {
        fn cb(_: void, response: zenai.gemini.types.GenerateContentResponse) void {
            if (response.text()) |t| {
                std.debug.print("{s}", .{t});
            }
        }
    }.cb,
);
```

</details>

<details>
<summary>Chat</summary>

```zig
var chat = zenai.gemini.Chat.init(&client, "gemini-2.5-flash", .{ .temperature = 0 }, .{});
defer chat.deinit();

const r1 = try chat.sendMessage("My name is Alice.");
std.debug.print("{s}\n", .{r1.text() orelse ""});

const r2 = try chat.sendMessage("What is my name?");
std.debug.print("{s}\n", .{r2.text() orelse ""});
```

</details>

<details>
<summary>Function calling</summary>

```zig
const tools = [_]zenai.gemini.types.Tool{.{
    .functionDeclarations = &.{.{
        .name = "get_weather",
        .description = "Get the current weather for a city.",
        .parameters = .{
            .type = .OBJECT,
            .properties = &.{
                .{ .key = "city", .value = .{ .type = .STRING } },
            },
            .required = &.{"city"},
        },
    }},
}};

var response = try client.generateContentFromText(
    "gemini-2.5-flash",
    "What's the weather in Paris?",
    .{},
    .{ .tools = &tools },
);
defer response.deinit();

if (response.value.firstFunctionCall()) |fc| {
    std.debug.print("Call: {s}\n", .{fc.name orelse ""});
}
```

</details>

<details>
<summary>Vertex AI</summary>

The same client can target [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs) instead of the Developer API, in one of two modes.

**Express mode** takes a plain [Vertex API key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview) and needs no project:

```zig
var client = zenai.vertex.Client.init(io, allocator, api_key, .{ .vertex = .{} });
defer client.deinit();
```

**Project/location mode** takes an OAuth access token as the key. Refreshing it is your job; tokens expire after about an hour.

```bash
export TOKEN=$(gcloud auth print-access-token)
```

```zig
var client = zenai.vertex.Client.init(io, allocator, token, .{
    .vertex = .{ .project = "my-project", .location = "global" },
});
defer client.deinit();
```

Generation, streaming, `Chat` and `countTokens` work in both modes. Model listing needs project/location mode, because Google's `ListPublisherModels` rejects API keys. Embeddings, file uploads and cached content are Developer-API-only and return `error.UnsupportedByBackend` on Vertex.

Through the provider abstraction, `.vertex` is picked when `GOOGLE_GENAI_USE_VERTEXAI=1` (or `true`) is set. Express mode reads `GOOGLE_API_KEY`. Project mode reads `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`/`GOOGLE_CLOUD_REGION` (default `"global"`), but you still pass the access token explicitly as the credential.

</details>

## OpenAI

Set your API key ([get one here](https://platform.openai.com/api-keys)):

```bash
export OPENAI_API_KEY='your-api-key'
```

```zig
const zenai = @import("zenai");

const api_key = environ.getPosix("OPENAI_API_KEY") orelse return error.MissingApiKey;
var client = zenai.openai.Client.init(io, allocator, api_key, .{});
defer client.deinit();

var response = try client.chatCompletionFromText("gpt-4o", "What is Zig?", .{});
defer response.deinit();

std.debug.print("{s}\n", .{response.value.text() orelse ""});
```

<details>
<summary>Streaming</summary>

```zig
try client.chatCompletionStreamFromText(
    "gpt-4o",
    "Write a poem about the moon.",
    .{},
    {},
    &struct {
        fn cb(_: void, response: zenai.openai.types.ChatCompletionResponse) void {
            if (response.text()) |t| {
                std.debug.print("{s}", .{t});
            }
        }
    }.cb,
);
```

</details>

<details>
<summary>Chat</summary>

```zig
var chat = zenai.openai.Chat.init(&client, "gpt-4o", .{ .temperature = 0 });
defer chat.deinit();

const r1 = try chat.sendMessage("My name is Alice.");
std.debug.print("{s}\n", .{r1.text() orelse ""});

const r2 = try chat.sendMessage("What is my name?");
std.debug.print("{s}\n", .{r2.text() orelse ""});
```

</details>

<details>
<summary>Function calling</summary>

```zig
const tools = [_]zenai.openai.types.Tool{.{
    .type = "function",
    .function = .{
        .name = "get_weather",
        .description = "Get the current weather for a city.",
    },
}};

var response = try client.chatCompletion("gpt-4o", &.{
    .{ .role = .user, .content = "What's the weather in Paris?" },
}, .{ .tools = &tools });
defer response.deinit();

if (response.value.firstToolCall()) |tc| {
    std.debug.print("Call: {s}\n", .{tc.function.?.name orelse ""});
}
```

</details>

## ChatGPT subscription (Codex)

`zenai.codex` talks to the backend Codex CLI uses (`chatgpt.com/backend-api/codex`), so requests count against a ChatGPT plan rather than API credits. It speaks the same Responses API as the OpenAI client but authenticates with the subscription's OAuth access token and the ChatGPT account id from that token's JWT.

zenai doesn't run the OAuth login. Get the token yourself (the browser's `lightpanda agent` does it in `src/agent/auth/codex.zig`), then go through the provider abstraction:

```zig
var codex_client = try zenai.codex.Client.init(io, allocator, access_token, .{
    .account_id = account_id,
});
defer codex_client.deinit();
const ai: zenai.provider.Client = .{ .codex = &codex_client };

var result = try ai.generateContent("gpt-5.5", &.{
    .{ .role = .user, .content = "What is Zig?" },
}, .{});
defer result.deinit();
```

The backend is streaming-only; `generateContent` accumulates the stream for you. Access tokens expire, so call `ai.setApiKey(new_token)` after a refresh. It keeps the open connections, and you free the old token buffer only after it returns.

## Anthropic

Set your API key ([get one here](https://console.anthropic.com/settings/keys)):

```bash
export ANTHROPIC_API_KEY='your-api-key'
```

```zig
const zenai = @import("zenai");

const api_key = environ.getPosix("ANTHROPIC_API_KEY") orelse return error.MissingApiKey;
var client = zenai.anthropic.Client.init(io, allocator, api_key, .{});
defer client.deinit();

var response = try client.createMessageFromText("claude-sonnet-4-6", "What is Zig?", 1024, .{});
defer response.deinit();

std.debug.print("{s}\n", .{response.value.text() orelse ""});
```

<details>
<summary>Streaming</summary>

```zig
try client.createMessageStreamFromText(
    "claude-sonnet-4-6",
    "Write a poem about the moon.",
    1024,
    .{},
    {},
    &struct {
        fn cb(_: void, event: zenai.anthropic.types.StreamEvent) void {
            if (event.delta) |delta| {
                if (delta.text) |t| {
                    std.debug.print("{s}", .{t});
                }
            }
        }
    }.cb,
);
```

</details>

<details>
<summary>Chat</summary>

```zig
var chat = zenai.anthropic.Chat.init(&client, "claude-sonnet-4-6", 1024, .{});
defer chat.deinit();

const r1 = try chat.sendMessage("My name is Alice.");
std.debug.print("{s}\n", .{r1.text() orelse ""});

const r2 = try chat.sendMessage("What is my name?");
std.debug.print("{s}\n", .{r2.text() orelse ""});
```

</details>

<details>
<summary>Function calling</summary>

```zig
const tools = [_]zenai.anthropic.types.Tool{.{
    .name = "get_weather",
    .description = "Get the current weather for a city.",
    .input_schema = // JSON Schema as std.json.Value
}};

var response = try client.createMessage("claude-sonnet-4-6", &.{
    .{ .role = .user, .content = &.{.{ .text = "What's the weather in Paris?" }} },
}, 1024, .{ .tools = &tools });
defer response.deinit();

if (response.value.firstToolUse()) |tu| {
    std.debug.print("Call: {s}\n", .{tu.name orelse ""});
}
```

</details>

## Search

Every search client follows the same shape: `init` with a key, call `search`, iterate the results. Tavily as an example:

```zig
const zenai = @import("zenai");

const api_key = zenai.search.tavily.Client.envApiKey(environ) orelse return error.MissingApiKey;
var client = zenai.search.tavily.Client.init(io, allocator, api_key, .{});
defer client.deinit();

var response = try client.search("what is zig", .{ .max_results = 5 });
defer response.deinit();

for (response.value.results) |r| {
    std.debug.print("{s} — {s}\n", .{ r.title, r.url });
}
```

| Provider | Namespace | Key (`Client.env_var_name`) | Results |
| --- | --- | --- | --- |
| [Tavily](https://app.tavily.com/) | `zenai.search.tavily` | `TAVILY_API_KEY` | `{title, url, content}` |
| [Brave](https://api-dashboard.search.brave.com/) | `zenai.search.brave` | `BRAVE_API_KEY` | `{title, url, description}` per section |
| [Exa](https://dashboard.exa.ai/) | `zenai.search.exa` | `EXA_API_KEY` | metadata, text only on request |
| [Keenable](https://keenable.ai/console) | `zenai.search.keenable` | `KEENABLE_API_KEY` (optional) | `{title, url, snippet}` |
| [Parallel](https://platform.parallel.ai/) | `zenai.search.parallel` | `PARALLEL_API_KEY` | `{title, url, excerpts}` |

The others differ in ways that will catch you out:

<details>
<summary>Brave: results are grouped by section</summary>

Results come back under sections such as `web`, each of which may be missing. `text_decorations = false` strips the `<strong>` highlight markup from descriptions.

```zig
var client = zenai.search.brave.Client.init(io, allocator, api_key, .{});
defer client.deinit();

var response = try client.search("what is zig", .{ .count = 5, .text_decorations = false });
defer response.deinit();

if (response.value.web) |web| {
    for (web.results) |r| {
        std.debug.print("{s} — {s}\n", .{ r.title, r.url });
    }
}
```

</details>

<details>
<summary>Exa: snippet text is opt-in</summary>

A bare Exa search returns metadata only. Request `contents` to get any text; highlights are the cheapest form. `title` is nullable because Exa returns null for pages it could not title.

```zig
var client = zenai.search.exa.Client.init(io, allocator, api_key, .{});
defer client.deinit();

var response = try client.search("what is zig", .{
    .numResults = 5,
    .contents = .{ .highlights = .{ .numSentences = 3 } },
});
defer response.deinit();

for (response.value.results) |r| {
    std.debug.print("{s} — {s}\n", .{ r.title orelse "(untitled)", r.url });
}
```

</details>

<details>
<summary>Keenable: works without a key</summary>

Pass `null` as the key and the client calls the public endpoint, rate-limited per client IP. A key lifts the limits. `app_title` has no default: it is sent as `X-Keenable-Title`, required when keyless, and only you know your app's name.

```zig
const api_key = zenai.search.keenable.Client.envApiKey(environ); // null is fine
var client = zenai.search.keenable.Client.init(io, allocator, api_key, .{ .app_title = "my-app" });
defer client.deinit();

var response = try client.search("what is zig", .{ .max_results = 5 });
defer response.deinit();

for (response.value.results) |r| {
    std.debug.print("{s} — {s}\n", .{ r.title, r.snippet });
}
```

</details>

<details>
<summary>Parallel: an objective plus keyword queries</summary>

Parallel picks excerpts for the request's `objective`, a natural-language statement of what you're looking for. The `query` argument is sent as the single keyword query; set `search_queries` to send several (the API recommends 2-3). `mode` trades quality for latency: `advanced` (the default, ~3s), `basic`, `fast` (~1s) or `turbo`.

```zig
var client = zenai.search.parallel.Client.init(io, allocator, api_key, .{});
defer client.deinit();

var response = try client.search("zig 0.17 release notes", .{
    .objective = "What changed in the Zig 0.17 standard library?",
    .mode = .fast,
    .advanced_settings = .{ .max_results = 5 },
});
defer response.deinit();

for (response.value.results) |r| {
    std.debug.print("{s} — {s}\n", .{ r.title orelse "(untitled)", r.url });
    for (r.excerpts) |e| std.debug.print("  {s}\n", .{e});
}
```

</details>

## TypeSafe (System One)

[Jev](https://docs.typesafe.ai) is a judgement model. You send the `state` under judgement plus a map of typed questions about it, and get one typed answer per question back:

- `noul`: a probability
- `choice`: a label, with per-option probabilities
- `score`: a level on an ordered scale

There is no free text and no tool calling. Set your API key ([get one here](https://typesafe.ai/)):

```bash
export TYPESAFE_API_KEY='your-api-key'
```

```zig
const zenai = @import("zenai");
const typesafe = zenai.typesafe.types;

const api_key = zenai.typesafe.envApiKey(environ) orelse return error.MissingApiKey;
var client: zenai.typesafe.Client = .init(io, allocator, api_key, .{});
defer client.deinit();

// The question ids are yours; the answers come back under the same ids.
const questions: typesafe.Questions = .init(&.{
    .{ .key = "is_complaint", .value = .noulText("Is the customer complaining?") },
    .{ .key = "topic", .value = .choiceText(
        "What is this message about?",
        typesafe.choices(&.{ "billing", "delivery", "other" }),
    ) },
    .{ .key = "anger", .value = .scoreText(
        "How angry is the customer?",
        typesafe.levels(&.{ "Calm", "Annoyed", "Furious" }),
    ) },
});
var response = try client.ask(.{ .text = "The pizza arrived cold and an hour late." }, questions, .{});
defer response.deinit(); // the answers borrow it

// Accessors by question id: null for a question the model left out or one
// asked for the wrong thing, rather than an unwrap that panics.
std.debug.print("complaint: {d:.2}\n", .{response.value.noul("is_complaint").?});
std.debug.print("topic: {s} ({d:.2} confident)\n", .{
    try response.value.choice("topic", questions), // validated against the criteria
    response.value.confidence("topic").?,
});
```

Jev ingests the state once and evaluates every question against it in parallel, so packing several questions into one request, including speculative ones you may discard, costs far less than one call each.

Things to know before relying on it:

- **Limits.** One request carries up to 64k tokens, and the state plus the longest single question must fit in 32k. An overrun comes back as `error.ApiError` with a 422 in `client.last_error`.
- **Validate choices.** `typesafe.validateChoice(answer, offered)` rejects an answer whose choice was never offered, whose probabilities don't cover exactly the offered set, or whose choice isn't the argmax. `response.choice(id, questions)` and `choiceEnum` run it for you.
- **Model versions.** `response.model` is the concrete version that answered, not the alias you asked for, and aliases move without notice. It borrows the response; use `response.dupeModel(allocator)` to keep it. `client.listModels()` lists the versions behind `jev-latest` and `jev-preview`.

<details>
<summary>Describing options and structured input</summary>

Options that need their own description are written out in full; `null` means the id speaks for itself:

```zig
.{ .key = "topic", .value = .{ .choice = .{
    .instructions = .{ .text = "What is this message about?" },
    .criteria = .init(&.{
        .{ .key = "billing", .value = null },
        .{ .key = "delivery", .value = .{ .text = "Anything about shipping or timing" } },
    }),
} } },
```

`instructions`, every criteria description, and `state` all accept `.{ .json = value }` in place of `.{ .text = "…" }`. That is how you judge a whole conversation or record in one call.

</details>

<details>
<summary>Choices from an enum</summary>

When the options are an enum you already have, `enumChoices` builds the criteria from its tags and `choiceEnum` maps the answer back, so the option set and the type that acts on it cannot drift apart:

```zig
const Verdict = enum { approve, review, reject };

fn describe(v: Verdict) []const u8 {
    return switch (v) {
        .approve => "Nothing here needs a human.",
        .review => "A human should look before this goes out.",
        .reject => "Refuse it outright.",
    };
}

const questions: typesafe.Questions = .init(&.{
    .{ .key = "verdict", .value = .choiceText(
        "What should happen to this message?",
        // A subset is normal: offer only what the caller will act on.
        try typesafe.enumChoices(arena, Verdict, &.{ .approve, .review }, describe),
    ) },
});
// …
const verdict = try response.value.choiceEnum("verdict", questions, Verdict);
```

</details>

## Features

| | Gemini | OpenAI | Anthropic |
| --- | :---: | :---: | :---: |
| Generation + SSE streaming | ✓ | ✓ | ✓ |
| Multi-turn chat | ✓ | ✓ | ✓ |
| Function calling | ✓ | ✓ | ✓ |
| Embeddings | ✓ | ✓ | |
| Model listing | ✓ | ✓ | |
| Token counting | ✓ | | |
| File uploads (resumable) | ✓ | | |
| Cached content | ✓ | | |
| Safety settings | ✓ | | |
| Extended thinking | | | ✓ |

The provider abstraction covers generation, streaming and embeddings across all of them, and `lastError()` returns the status and message behind a failed request.

Search options beyond the basics:

- **Tavily**: synthesized answers, domain include/exclude, news/general topic, time range
- **Brave**: country and language targeting, safesearch, freshness, section filtering, extra snippets
- **Exa**: search type, category, domain include/exclude, published-date windows, contents (highlights, text, summary)
- **Keenable**: site restriction, published-date windows, snippet-length hint
- **Parallel**: objective, multiple keyword queries, latency mode, domain include/exclude, after-date filter, live fetch policy, excerpt-length caps, location

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
