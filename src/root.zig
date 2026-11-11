pub const gemini = struct {
    pub const Client = @import("gemini/Client.zig");
    pub const types = @import("gemini/types.zig");
    pub const decisions = @import("gemini/decisions.zig");
};

pub const openai = struct {
    pub const Client = @import("openai/Client.zig");
    pub const types = @import("openai/types.zig");
    /// Native Ollama `/api/chat` (so `num_ctx` can be set).
    pub const ollama = @import("openai/ollama.zig");
};

pub const anthropic = struct {
    pub const Client = @import("anthropic/Client.zig");
    pub const types = @import("anthropic/types.zig");
};

/// OpenAI Codex — the ChatGPT-subscription backend. Reuses the openai Responses
/// types; see `codex/Client.zig`.
pub const codex = struct {
    pub const Client = @import("codex/Client.zig");
};

/// Search providers — separate namespace from the LLM clients above.
/// Serper/Google CSE could land as siblings here.
pub const search = struct {
    pub const tavily = struct {
        pub const Client = @import("search/tavily/Client.zig");
        pub const types = @import("search/tavily/types.zig");
    };
    pub const brave = struct {
        pub const Client = @import("search/brave/Client.zig");
        pub const types = @import("search/brave/types.zig");
    };
    pub const exa = struct {
        pub const Client = @import("search/exa/Client.zig");
        pub const types = @import("search/exa/types.zig");
    };
    pub const keenable = struct {
        pub const Client = @import("search/keenable/Client.zig");
        pub const types = @import("search/keenable/types.zig");
    };
    pub const parallel = struct {
        pub const Client = @import("search/parallel/Client.zig");
        pub const types = @import("search/parallel/types.zig");
    };
};

/// TypeSafe System One — the Jev judgement model. It answers typed questions
/// rather than generating text, so it is not part of `provider.Client`.
pub const typesafe = struct {
    pub const Client = @import("typesafe/Client.zig");
    pub const types = @import("typesafe/types.zig");
    pub const env_var_name = Client.env_var_name;
    pub const base_url_env_var_name = Client.base_url_env_var_name;
};

/// Decision models, from whichever provider has a key: TypeSafe System One
/// (Jev) or Gemini's Decisions API. Both answer typed questions (`noul`,
/// `choice`, `score`) about one input with calibrated probabilities rather
/// than generating text. Each answer is a judgment; the calling code makes
/// the decision. The question and answer types are TypeSafe's, which can
/// hold everything Gemini's can; `Pool` converts for Gemini.
pub const decisions = struct {
    pub const Pool = @import("decisions/Pool.zig");
    /// Questions from JSON and answers back as JSON, for a caller driven by
    /// configuration rather than comptime.
    pub const json = @import("decisions/json.zig");
    pub const gemini = @import("decisions/gemini.zig");
    pub const types = typesafe.types;

    pub const Provider = Pool.Provider;
    pub const AskOptions = Pool.AskOptions;
    pub const Response = http.Response(types.AskResponse);
    pub const envCredentials = Pool.envCredentials;
    pub const envProvider = Pool.envProvider;
    pub const env_hint = Pool.env_hint;
    pub const gemini_base_url_env_var_name = Pool.gemini_base_url_env_var_name;

    /// Shorthands for building questions.
    pub const Content = types.Content;
    pub const Question = types.Question;
    pub const Questions = types.Questions;
    pub const QuestionEntry = types.QuestionEntry;
    pub const ChoiceEntry = types.ChoiceCriteria.Entry;
    pub const enumChoices = types.enumChoices;
};

pub const provider = @import("provider.zig");
/// Multi-turn conversation on `provider.Client`, for any backend.
pub const Chat = @import("Chat.zig");
pub const retry = @import("retry.zig");
pub const json = @import("json.zig");
pub const http = @import("http.zig");

const std = @import("std");

/// Recursively reference every public decl (the removed
/// `std.testing.refAllDeclsRecursive`) so API breakage in paths no unit
/// test exercises still fails `zig build test`.
fn refAllDeclsRecursive(comptime T: type) void {
    inline for (comptime std.meta.declarations(T)) |name| {
        if (@TypeOf(@field(T, name)) == type) {
            switch (@typeInfo(@field(T, name))) {
                .@"struct", .@"enum", .@"union", .@"opaque" => refAllDeclsRecursive(@field(T, name)),
                else => {},
            }
        }
        _ = &@field(T, name);
    }
}

test {
    refAllDeclsRecursive(@This());
}
