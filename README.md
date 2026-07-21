# zenai

Zig client for AI APIs, supporting [Google Gemini](https://ai.google.dev/gemini-api/docs) (Developer API and [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs)), [OpenAI](https://platform.openai.com/docs/api-reference), and [Anthropic](https://docs.anthropic.com/en/docs/about-claude/models). OpenAI-compatible endpoints — [Ollama](https://github.com/ollama/ollama/blob/main/docs/openai.md), [Hugging Face Inference](https://huggingface.co/docs/inference-providers/index), and [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) (`llama-server`) — are supported through the OpenAI client. Ported from the official [Go Gen AI SDK](https://github.com/googleapis/go-genai), [openai-go](https://github.com/openai/openai-go), and [anthropic-sdk-go](https://github.com/anthropics/anthropic-sdk-go). Also ships an `agent infrastructure` namespace under `zenai.search` — currently [Tavily](https://docs.tavily.com/), with room for sibling providers.

<img width="1024" height="1024" alt="Meditating panda with incense smoke" src="https://github.com/user-attachments/assets/b9c82960-05ec-4aa1-b171-092ee2126551" />


## Installation

```bash
zig fetch --save git+https://github.com/lightpanda-io/zenai
```

Then add the dependency in your `build.zig`:

```zig
const zenai = b.dependency("zenai", .{});
exe.root_module.addImport("zenai", zenai.module("zenai"));
```

Requires Zig >= 0.16.0. The examples below assume `allocator`, `io`, and `environ` are in scope; with Zig 0.16's main signature they come straight from `std.process.Init`:

```zig
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const environ = init.minimal.environ;
    // ...
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
var client = zenai.gemini.Client.init(allocator, io, api_key, .{});
defer client.deinit();

var response = try client.generateContentFromText("gemini-2.5-flash", "What is Zig?", .{}, .{});
defer response.deinit();

std.debug.print("{s}\n", .{response.value.text() orelse ""});
```

### Streaming

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

### Chat

```zig
var chat = zenai.gemini.Chat.init(&client, "gemini-2.5-flash", .{ .temperature = 0 }, .{});
defer chat.deinit();

const r1 = try chat.sendMessage("My name is Alice.");
std.debug.print("{s}\n", .{r1.text() orelse ""});

const r2 = try chat.sendMessage("What is my name?");
std.debug.print("{s}\n", .{r2.text() orelse ""});
```

### Function calling

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

### Google Vertex AI

The same Gemini client can target [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs) instead of the Gemini Developer API. Two modes:

**Express mode** — a plain [Vertex API key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview), no project needed:

```zig
var client = zenai.vertex.Client.init(allocator, io, api_key, .{ .vertex = .{} });
defer client.deinit();
```

**Project/location mode** — pass an OAuth access token as the key (refreshing it is your job; tokens expire after ~1 hour):

```bash
export TOKEN=$(gcloud auth print-access-token)
```

```zig
var client = zenai.vertex.Client.init(allocator, io, token, .{
    .vertex = .{ .project = "my-project", .location = "global" },
});
defer client.deinit();
```

Generation, streaming, `Chat`, and `countTokens` work in both modes. Model listing works on Vertex too, but Google's `ListPublisherModels` rejects API keys, so it needs project/location mode (OAuth). Embeddings, file uploads, and cached content are Developer-API-only and return `error.UnsupportedByBackend` on Vertex.

Through the provider abstraction, `.vertex` is env-detected when `GOOGLE_GENAI_USE_VERTEXAI=1` (or `true`) is set: express mode reads `GOOGLE_API_KEY`; project mode reads `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`/`GOOGLE_CLOUD_REGION` (default `"global"`) but needs the access token passed explicitly as the credential.

## OpenAI

Set your API key ([get one here](https://platform.openai.com/api-keys)):

```bash
export OPENAI_API_KEY='your-api-key'
```

```zig
const zenai = @import("zenai");

const api_key = environ.getPosix("OPENAI_API_KEY") orelse return error.MissingApiKey;
var client = zenai.openai.Client.init(allocator, io, api_key, .{});
defer client.deinit();

var response = try client.chatCompletionFromText("gpt-4o", "What is Zig?", .{});
defer response.deinit();

std.debug.print("{s}\n", .{response.value.text() orelse ""});
```

### Streaming

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

### Chat

```zig
var chat = zenai.openai.Chat.init(&client, "gpt-4o", .{ .temperature = 0 });
defer chat.deinit();

const r1 = try chat.sendMessage("My name is Alice.");
std.debug.print("{s}\n", .{r1.text() orelse ""});

const r2 = try chat.sendMessage("What is my name?");
std.debug.print("{s}\n", .{r2.text() orelse ""});
```

### Function calling

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

## Anthropic

Set your API key ([get one here](https://console.anthropic.com/settings/keys)):

```bash
export ANTHROPIC_API_KEY='your-api-key'
```

```zig
const zenai = @import("zenai");

const api_key = environ.getPosix("ANTHROPIC_API_KEY") orelse return error.MissingApiKey;
var client = zenai.anthropic.Client.init(allocator, io, api_key, .{});
defer client.deinit();

var response = try client.createMessageFromText("claude-sonnet-4-6", "What is Zig?", 1024, .{});
defer response.deinit();

std.debug.print("{s}\n", .{response.value.text() orelse ""});
```

### Streaming

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

### Chat

```zig
var chat = zenai.anthropic.Chat.init(&client, "claude-sonnet-4-6", 1024, .{});
defer chat.deinit();

const r1 = try chat.sendMessage("My name is Alice.");
std.debug.print("{s}\n", .{r1.text() orelse ""});

const r2 = try chat.sendMessage("What is my name?");
std.debug.print("{s}\n", .{r2.text() orelse ""});
```

### Function calling

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

## Tavily (search)

Tavily is an AI-friendly search API that returns clean `{title, url, content}` JSON results — handy as a low-noise alternative to scraping a SERP. Set your API key ([get one here](https://app.tavily.com/)):

```bash
export TAVILY_API_KEY='tvly-...'
```

```zig
const zenai = @import("zenai");

const api_key = environ.getPosix("TAVILY_API_KEY") orelse return error.MissingApiKey;
var client = zenai.search.tavily.Client.init(allocator, io, api_key, .{});
defer client.deinit();

var response = try client.search("what is zig", .{ .max_results = 5 });
defer response.deinit();

for (response.value.results) |r| {
    std.debug.print("{s} — {s}\n", .{ r.title, r.url });
}
```

## Provider Abstraction

Use `zenai.provider.Client` to write provider-agnostic code. Swap providers by changing one line:

```zig
const zenai = @import("zenai");

// Pick your provider:
var gemini_client = zenai.gemini.Client.init(allocator, io, gemini_key, .{});
defer gemini_client.deinit();
const ai: zenai.provider.Client = .{ .gemini = &gemini_client };

// Or Vertex AI (same Gemini client, Vertex backend — see the Vertex section):
// var vertex_client = zenai.vertex.Client.init(allocator, io, token, .{
//     .vertex = .{ .project = "my-project" },
// });
// const ai: zenai.provider.Client = .{ .vertex = &vertex_client };

// Or:
// var openai_client = zenai.openai.Client.init(allocator, io, openai_key, .{});
// const ai: zenai.provider.Client = .{ .openai = &openai_client };

// Or:
// var anthropic_client = zenai.anthropic.Client.init(allocator, io, anthropic_key, .{});
// const ai: zenai.provider.Client = .{ .anthropic = &anthropic_client };

// Or Hugging Face (OpenAI-compatible). Token comes from HF_TOKEN. Defaults to the
// serverless router; pass `.base_url` for a dedicated Inference Endpoint:
// var hf_client = zenai.huggingface.Client.init(allocator, io, hf_token, .{
//     .base_url = "https://router.huggingface.co/v1",
// });
// const ai: zenai.provider.Client = .{ .huggingface = &hf_client };

// Or a local llama.cpp `llama-server` (OpenAI-compatible). No key needed;
// defaults to http://localhost:8080/v1 — override with `.base_url`:
// var llama_client = zenai.llama_cpp.Client.init(allocator, io, "llama.cpp", .{
//     .base_url = "http://localhost:8080/v1",
// });
// const ai: zenai.provider.Client = .{ .llama_cpp = &llama_client };

var result = try ai.generateContent("gemini-2.5-flash", &.{
    .{ .role = .user, .content = "What is Zig?" },
}, .{});
defer result.deinit();

std.debug.print("{s}\n", .{result.text orelse ""});
```

Drop down to provider-specific APIs when needed:

```zig
switch (ai) {
    .gemini => |g| {
        // Use Gemini-specific features like cached content, file uploads, etc.
        var cached = try g.createCachedContent("gemini-2.5-flash", .{ ... });
    },
    else => {},
}
```

## Features

**Gemini:**
- Text generation and streaming (SSE)
- Multi-turn chat with history management
- Function calling and tool use
- Embeddings
- Token counting
- File uploads (resumable protocol)
- Cached content
- Model listing and info
- Safety settings and content filtering
- Vertex AI backend (express-mode API key, or project/location with an OAuth access token) for generation, streaming, chat, token counting, and model listing

**OpenAI:**
- Chat completions and streaming (SSE)
- Multi-turn chat with history management
- Function calling and tool use
- Embeddings
- Model listing and info

**Anthropic:**
- Message creation and streaming (SSE)
- Multi-turn chat with history management
- Function calling and tool use
- Extended thinking support

**Search providers:**
- Tavily (`zenai.search.tavily`) — JSON search API with optional synthesized answers, domain include/exclude, news/general topic, time-range filtering

**Provider abstraction:**
- Unified text generation, streaming, and embeddings
- OpenAI-compatible backends: Ollama, Hugging Face, and llama.cpp (`llama-server`)
- `lastError()` to surface the status and message behind a failed request
- Escape hatches to provider-specific APIs

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
