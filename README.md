# zenai

Zig client for AI APIs, supporting [Google Gemini](https://ai.google.dev/gemini-api/docs) and [OpenAI](https://platform.openai.com/docs/api-reference). Ported from the official [Go Gen AI SDK](https://github.com/googleapis/go-genai) and [openai-go](https://github.com/openai/openai-go).

## Installation

```bash
zig fetch --save git+https://github.com/lightpanda-io/zenai
```

Then add the dependency in your `build.zig`:

```zig
const zenai = b.dependency("zenai", .{});
exe.root_module.addImport("zenai", zenai.module("zenai"));
```

## Gemini

Set your API key ([get one here](https://ai.google.dev/gemini-api/docs/api-key)):

```bash
export GOOGLE_API_KEY='your-api-key'
```

```zig
const zenai = @import("zenai");

const api_key = std.posix.getenv("GOOGLE_API_KEY") orelse return error.MissingApiKey;
var client = zenai.gemini.Client.init(allocator, api_key, .{});
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
                const fd = std.posix.STDOUT_FILENO;
                _ = std.posix.write(fd, t) catch return;
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

## OpenAI

Set your API key ([get one here](https://platform.openai.com/api-keys)):

```bash
export OPENAI_API_KEY='your-api-key'
```

```zig
const zenai = @import("zenai");

const api_key = std.posix.getenv("OPENAI_API_KEY") orelse return error.MissingApiKey;
var client = zenai.openai.Client.init(allocator, api_key, .{});
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
                const fd = std.posix.STDOUT_FILENO;
                _ = std.posix.write(fd, t) catch return;
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

## Provider Abstraction

Use `zenai.provider.Client` to write provider-agnostic code. Swap providers by changing one line:

```zig
const zenai = @import("zenai");

// Pick your provider:
var gemini_client = zenai.gemini.Client.init(allocator, gemini_key, .{});
defer gemini_client.deinit();
const ai: zenai.provider.Client = .{ .gemini = &gemini_client };

// Or:
// var openai_client = zenai.openai.Client.init(allocator, openai_key, .{});
// const ai: zenai.provider.Client = .{ .openai = &openai_client };

var result = try ai.generateContent("gemini-2.5-flash", &.{
    .{ .role = .user, .content = "What is Zig?" },
}, .{});
defer result.deinit();

std.debug.print("{s}\n", .{result.text orelse ""});
```

Drop down to provider-specific APIs when needed:

```zig
if (ai.asGemini()) |g| {
    // Use Gemini-specific features like cached content, file uploads, etc.
    var cached = try g.createCachedContent("gemini-2.5-flash", .{ ... });
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

**OpenAI:**
- Chat completions and streaming (SSE)
- Multi-turn chat with history management
- Function calling and tool use
- Embeddings
- Model listing and info

**Provider abstraction:**
- Unified text generation, streaming, and embeddings
- Escape hatches to provider-specific APIs

## License

GNU Affero General Public License v3.0 — see [LICENSE](LICENSE) for details.
