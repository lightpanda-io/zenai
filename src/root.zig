pub const gemini = struct {
    pub const Client = @import("gemini/Client.zig");
    pub const Chat = @import("gemini/Chat.zig");
    pub const types = @import("gemini/types.zig");
};

pub const openai = struct {
    pub const Client = @import("openai/Client.zig");
    pub const Chat = @import("openai/Chat.zig");
    pub const types = @import("openai/types.zig");
    /// Native Ollama `/api/chat` (so `num_ctx` can be set).
    pub const ollama = @import("openai/ollama.zig");
};

pub const anthropic = struct {
    pub const Client = @import("anthropic/Client.zig");
    pub const Chat = @import("anthropic/Chat.zig");
    pub const types = @import("anthropic/types.zig");
};

/// Ollama uses the OpenAI-compatible API with a different default base URL.
pub const ollama = openai;

/// Hugging Face Inference uses the OpenAI-compatible Chat Completions API with a
/// different default base URL — the serverless router, or a dedicated Inference
/// Endpoint passed via `base_url`.
pub const huggingface = openai;

/// Search providers — separate namespace from the LLM clients above. Tavily
/// is the first; Brave/Serper/Google CSE could land as siblings here.
pub const search = struct {
    pub const tavily = struct {
        pub const Client = @import("search/tavily/Client.zig");
        pub const types = @import("search/tavily/types.zig");
    };
};

pub const provider = @import("provider.zig");
pub const retry = @import("retry.zig");
pub const json = @import("json.zig");
pub const http = @import("http.zig");

test {
    _ = gemini.Client;
    _ = gemini.Chat;
    _ = gemini.types;
    _ = openai.Client;
    _ = openai.Chat;
    _ = openai.types;
    _ = openai.ollama;
    _ = anthropic.Client;
    _ = anthropic.Chat;
    _ = anthropic.types;
    _ = search.tavily.Client;
    _ = search.tavily.types;
    _ = provider;
    _ = retry;
    _ = json;
    _ = http;
}
