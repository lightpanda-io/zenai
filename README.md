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
- **Judgement** (`zenai.typesafe`): [TypeSafe System One](https://docs.typesafe.ai), whose Jev model answers typed questions about a piece of state (a probability, a labelled choice or a score) instead of generating text. `zenai.decisions` asks the same questions of Jev or Gemini's Decisions API, whichever has a key

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
const ai = try zenai.provider.Client.init(io, allocator, .llama_cpp, "llama.cpp", .{});
defer ai.deinit(allocator);
```

The first argument after the allocator is a `provider.Backend`: a tag, plus the settings only that provider takes (`.{ .vertex = .{ .project = "p" } }`, `.{ .codex = .{ .account_id = id } }`). The options struct holds what every provider takes: `base_url`, `retry_policy`, `request_timeout_ms`, `http_client` and `extra_headers` (sent on every request, e.g. Hugging Face's `X-HF-Bill-To`).

`init` reads no environment variables. `zenai.provider.fromEnv(environ, tag)` returns what the conventional ones configure for a tag: its key, its backend settings and, for `.openai_compatible`, the base URL:

```zig
const cfg = zenai.provider.fromEnv(environ, .openai_compatible);
const ai = try zenai.provider.Client.init(io, allocator, cfg.backend, cfg.key orelse return error.MissingApiKey, .{
    .base_url = cfg.base_url,
});
defer ai.deinit(allocator);
```

`zenai.provider.info(tag)` gives each tag's default base URL, key variable, default model and whether it is a local server.

Gateway model ids are provider-prefixed (`"anthropic/claude-sonnet-5"`). For Hugging Face, pass `.base_url` to target a dedicated Inference Endpoint instead of the serverless router. `provider.detectKeys` picks `.openai_compatible` instead of `.openai` when `OPENAI_BASE_URL` is set, which covers vLLM, LiteLLM, Together, Groq and the like.

When you need something the abstraction doesn't expose, switch on the tag and use the provider client directly:

```zig
switch (ai) {
    .gemini => |g| {
        // Gemini-specific features such as token counting.
        var count = try g.countTokensFromText("gemini-2.5-flash", "What is Zig?");
        defer count.deinit();
    },
    else => {},
}
```

### Chat

`zenai.Chat` keeps a conversation's history on top of any `provider.Client`. It copies everything it is given, and a failed turn leaves the history as it was.

```zig
var chat: zenai.Chat = .init(allocator, ai, "gemini-2.5-flash", .{ .system = "Be brief." });
defer chat.deinit();

_ = try chat.send("My name is Alice.");
const reply = try chat.send("What is my name?");
std.debug.print("{s}\n", .{reply.text orelse ""});
```

`sendStream` forwards the reply's text as it arrives, `sendParts` sends images, and `sendWithTools` runs the tool loop with its calls and results kept in the history.

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
<summary>Decisions</summary>

The Decisions API evaluates one shared `input` against a map of typed questions and returns a calibrated answer per question id: `noul` (a probability), `choice` (a label with per-option probabilities) or `score` (a level on an ordered scale). The input is ingested once, so put every question in the same request.

```zig
const decisions = zenai.gemini.decisions;

const questions: decisions.Questions = .init(&.{
    .{ .key = "is_complaint", .value = .noulText("Is the customer complaining?") },
    .{ .key = "topic", .value = .choiceText(
        "What is this message about?",
        decisions.choices(&.{ "billing", "delivery", "other" }),
    ) },
    .{ .key = "anger", .value = .scoreText("How angry is the customer?", &.{ "Calm", "Annoyed", "Furious" }) },
});
var response = try client.createDecision(decisions.default_model, .{ .text = "The pizza arrived cold." }, questions);
defer response.deinit(); // the answers borrow it

std.debug.print("complaint: {d:.2}\n", .{response.value.noul("is_complaint").?});
std.debug.print("topic: {s}\n", .{try response.value.choice("topic", questions)}); // validated against the criteria
```

The input can also be a list of content blocks (`.{ .parts = &.{ .{ .type = .image, .uri = file_uri, .mime_type = "image/png" } } }`). A blocked input comes back with empty `answers` and `prompt_feedback.block_reason` set.

</details>

<details>
<summary>Vertex AI</summary>

The same client can target [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs) instead of the Developer API, in one of two modes.

**Express mode** takes a plain [Vertex API key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview) and needs no project:

```zig
var client = zenai.gemini.Client.init(io, allocator, api_key, .{ .vertex = .{} });
defer client.deinit();
```

**Project/location mode** takes an OAuth access token as the key. Refreshing it is your job; tokens expire after about an hour.

```bash
export TOKEN=$(gcloud auth print-access-token)
```

```zig
var client = zenai.gemini.Client.init(io, allocator, token, .{
    .vertex = .{ .project = "my-project", .location = "global" },
});
defer client.deinit();
```

Generation, streaming, `Chat` and `countTokens` work in both modes. Model listing needs project/location mode, because Google's `ListPublisherModels` rejects API keys. Embeddings and decisions are Developer-API-only and return `error.UnsupportedByBackend` on Vertex.

Through the provider abstraction, the backend is `.{ .vertex = .{ .project = "my-project", .location = "global" } }`, or plain `.vertex` for express mode. `provider.detectKeys` picks `.vertex` when `GOOGLE_GENAI_USE_VERTEXAI=1` (or `true`) is set, and express mode reads `GOOGLE_API_KEY`. `provider.fromEnv(environ, .vertex)` fills project mode from `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`/`GOOGLE_CLOUD_REGION` (default `"global"`), but you still pass the access token explicitly as the credential.

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

The `.codex` provider talks to the backend Codex CLI uses (`chatgpt.com/backend-api/codex`), so requests count against a ChatGPT plan rather than API credits. It is the OpenAI client set up with `initChatGpt`: the same Responses API, authenticated with the subscription's OAuth access token and the ChatGPT account id from that token's JWT.

zenai doesn't run the OAuth login. Get the token yourself (the browser's `lightpanda agent` does it in `src/agent/auth/codex.zig`), then go through the provider abstraction:

```zig
const ai = try zenai.provider.Client.init(io, allocator, .{ .codex = .{ .account_id = account_id } }, access_token, .{});
defer ai.deinit(allocator);

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

- **Limits.** One request carries up to 64k tokens, and the state plus the longest single question must fit in 32k. An overrun comes back as `error.ApiError` with a 422 in `client.transport.last_error`.
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

### Either provider: `zenai.decisions`

Gemini's [Decisions API](#gemini) takes the same three question types. `zenai.decisions.Pool` asks TypeSafe, or Gemini when the caller's provider variable names it. Its questions and answers are its own provider-neutral types in `zenai.decisions.types`, shaped like the TypeSafe ones above (with the same `choices`, `levels`, `enumChoices`, `choice` and `choiceEnum` helpers), and `decisions.typesafe` and `decisions.gemini` convert them to and from each wire schema (Gemini gets JSON content as compact JSON text). It reads the environment on every ask, so exporting a key mid-run takes effect without a restart, and it keeps one warm client per concurrent caller:

```zig
var pool: zenai.decisions.Pool = .init(io, allocator, environFn, .{
    // Optional: lets MY_DECISION_PROVIDER=gemini pick Gemini over Jev.
    .provider_env_var_name = "MY_DECISION_PROVIDER",
});
defer pool.deinit();

const questions: zenai.decisions.Questions = .init(&.{
    .{ .key = "is_complaint", .value = .noulText("Is the customer complaining?") },
});
var failure: zenai.http.ErrorDetail = .{};
var response = try pool.ask(arena, .{ .text = "The pizza arrived cold." }, questions, .{}, &failure);
defer response.deinit();
```

`TYPESAFE_API_KEY` answers by default. Gemini (`GOOGLE_API_KEY`/`GEMINI_API_KEY`, not on Vertex) is only asked when the provider variable says `gemini`: a Google key is set for many other reasons, so having one is no consent to send content to a decision model. `envHint(name)` spells out what to set for a missing-key message. `GOOGLE_GEMINI_BASE_URL` and `TYPESAFE_BASE_URL` point either one elsewhere. A Gemini input blocked by safety filters fails with `error.InputBlocked`, with the reason in `failure`.

## Features

| | Gemini | OpenAI | Anthropic |
| --- | :---: | :---: | :---: |
| Generation + SSE streaming | ✓ | ✓ | ✓ |
| Function calling | ✓ | ✓ | ✓ |
| Embeddings | ✓ | ✓ | |
| Model listing | ✓ | ✓ | |
| Token counting | ✓ | | |
| Decisions (calibrated judgements) | ✓ | | |
| Safety settings | ✓ | | |
| Extended thinking | | | ✓ |

The provider abstraction covers generation, streaming, embeddings and multi-turn chat (`zenai.Chat`) across all of them, and `lastError()` returns the status and message behind a failed request.

Each client opens its own connections unless it is given one to borrow. To share one connection pool and CA bundle across clients (and threads, if its allocator is thread-safe), build a `std.http.Client` with `zenai.http.newClient(io, allocator)` and pass a pointer to it as `http_client` in each client's `InitOptions`, `provider.Client.InitOptions` or `decisions.Pool.Options`. It must outlive them, and you `deinit` it yourself.

Search options beyond the basics:

- **Tavily**: synthesized answers, domain include/exclude, news/general topic, time range
- **Brave**: country and language targeting, safesearch, freshness, section filtering, extra snippets
- **Exa**: search type, category, domain include/exclude, published-date windows, contents (highlights, text, summary)
- **Keenable**: site restriction, published-date windows, snippet-length hint
- **Parallel**: objective, multiple keyword queries, latency mode, domain include/exclude, after-date filter, live fetch policy, excerpt-length caps, location

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
