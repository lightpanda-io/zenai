const std = @import("std");
const zenai = @import("zenai");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const api_key = std.posix.getenv("GOOGLE_API_KEY") orelse
        std.posix.getenv("GEMINI_API_KEY") orelse
        {
            std.debug.print("Error: set GOOGLE_API_KEY or GEMINI_API_KEY environment variable\n", .{});
            std.process.exit(1);
        };

    var client = zenai.gemini.Client.init(allocator, api_key, .{});
    defer client.deinit();

    // --- Model info ---
    std.debug.print("=== Model info ===\n", .{});
    {
        var model_info = try client.getModel("gemini-2.5-flash");
        defer model_info.deinit();
        const m = model_info.value;
        std.debug.print("Name: {s}\n", .{m.name orelse "?"});
        std.debug.print("Display: {s}\n", .{m.displayName orelse "?"});
        std.debug.print("Input limit: {d} tokens\n", .{m.inputTokenLimit orelse 0});
        std.debug.print("Output limit: {d} tokens\n\n", .{m.outputTokenLimit orelse 0});
    }

    // --- Token counting ---
    std.debug.print("=== Token counting ===\n", .{});
    {
        var result = try client.countTokensFromText("gemini-2.5-flash", "What is the meaning of life?");
        defer result.deinit();
        std.debug.print("Tokens: {d}\n\n", .{result.value.totalTokens orelse 0});
    }

    // --- Embeddings ---
    std.debug.print("=== Embeddings ===\n", .{});
    {
        var result = try client.embedText("gemini-embedding-001", "What is the meaning of life?");
        defer result.deinit();
        if (result.value.embedding) |embedding| {
            if (embedding.values) |values| {
                std.debug.print("Dimension: {d}, first 3: [{d:.4}, {d:.4}, {d:.4}]\n\n", .{
                    values.len, values[0], values[1], values[2],
                });
            }
        }
    }

    // --- File upload ---
    std.debug.print("=== File upload ===\n", .{});
    {
        const text_data = "This is a test file for the Gemini API. It contains some sample text.";
        var result = try client.uploadFile(text_data, .{
            .displayName = "test.txt",
            .mimeType = "text/plain",
        });
        defer result.deinit();

        if (result.value.file) |file| {
            std.debug.print("Uploaded: {s}\n", .{file.name orelse "?"});
            std.debug.print("State: {s}\n", .{if (file.state) |s| @tagName(s) else "?"});
            std.debug.print("URI: {s}\n\n", .{file.uri orelse "?"});

            // Clean up — delete the test file
            client.deleteFile(file.name orelse "") catch {};
        }
    }

    // --- Cached content ---
    std.debug.print("=== Cached content ===\n", .{});
    {
        // List existing cached contents
        var list_result = try client.listCachedContents(.{});
        defer list_result.deinit();
        const count = if (list_result.value.cachedContents) |cc| cc.len else 0;
        std.debug.print("Existing cached contents: {d}\n\n", .{count});
    }

    // --- Simple text generation ---
    std.debug.print("=== Simple text generation ===\n", .{});
    {
        var response = try client.generateContentFromText(
            "gemini-2.5-flash",
            "What is your name?",
            .{ .temperature = 0 },
            .{},
        );
        defer response.deinit();

        if (response.value.text()) |text| {
            std.debug.print("{s}\n\n", .{text});
        }
    }

    // --- Streaming ---
    std.debug.print("=== Streaming ===\n", .{});
    try client.generateContentStreamFromText(
        "gemini-2.5-flash",
        "Write a short poem about the moon.",
        .{ .temperature = 0.7, .thinkingConfig = .{ .thinkingBudget = 0 } },
        .{},
        {},
        &printStreamChunk,
    );
    std.debug.print("\n\n", .{});

    // --- Chat session ---
    std.debug.print("=== Chat session ===\n", .{});
    {
        var chat = zenai.gemini.Chat.init(&client, "gemini-2.5-flash", .{
            .temperature = 0,
            .thinkingConfig = .{ .thinkingBudget = 0 },
        }, .{});
        defer chat.deinit();

        const r1 = try chat.sendMessage("My name is Alice. Remember it.");
        std.debug.print("User: My name is Alice. Remember it.\n", .{});
        std.debug.print("Model: {s}\n", .{r1.text() orelse "(no text)"});

        const r2 = try chat.sendMessage("What is my name?");
        std.debug.print("User: What is my name?\n", .{});
        std.debug.print("Model: {s}\n", .{r2.text() orelse "(no text)"});

        // Streaming chat — model still remembers context
        std.debug.print("User (stream): Tell me a joke about my name.\n", .{});
        std.debug.print("Model: ", .{});
        try chat.sendMessageStream("Tell me a joke about my name.", {}, &printStreamChunk);
        std.debug.print("\n\n", .{});
    }

    // --- Function calling ---
    std.debug.print("=== Function calling ===\n", .{});
    try functionCallingExample(&client);
}

fn printStreamChunk(_: void, response: zenai.gemini.types.GenerateContentResponse) void {
    if (response.text()) |t| {
        const fd = std.posix.STDOUT_FILENO;
        _ = std.posix.write(fd, t) catch return;
    }
}

fn functionCallingExample(client: *zenai.gemini.Client) !void {
    const model = "gemini-2.5-flash";

    // Define the tool
    const tools = [_]zenai.gemini.types.Tool{.{
        .functionDeclarations = &.{.{
            .name = "get_weather",
            .description = "Get the current weather for a given city.",
            .parameters = .{
                .type = .OBJECT,
                .properties = &.{
                    .{ .key = "city", .value = .{ .type = .STRING, .description = "The city name" } },
                },
                .required = &.{"city"},
            },
        }},
    }};

    const request_options = zenai.gemini.Client.RequestOptions{
        .tools = &tools,
    };

    // Step 1: Send the user prompt
    std.debug.print("User: What's the weather like in Paris?\n", .{});

    const user_parts = [_]zenai.gemini.types.Part{.{ .text = "What's the weather like in Paris?" }};
    const user_content = [_]zenai.gemini.types.Content{.{ .role = "user", .parts = &user_parts }};

    var response1 = try client.generateContent(
        model,
        &user_content,
        .{ .temperature = 0 },
        request_options,
    );
    defer response1.deinit();

    // Step 2: Check if the model wants to call a function
    const fc = response1.value.firstFunctionCall() orelse {
        std.debug.print("Model responded with text: {s}\n", .{response1.value.text() orelse "no text"});
        return;
    };

    std.debug.print("Model wants to call: {s}\n", .{fc.name orelse "unknown"});

    // Step 3: "Execute" the function (simulated)
    const weather_result = "{\"temperature\": \"22°C\", \"condition\": \"Sunny\", \"humidity\": \"45%\"}";
    std.debug.print("Function result: {s}\n", .{weather_result});

    // Step 4: Send the function response back to the model
    // Build the conversation history: user message + model's function call + function response
    const model_parts = response1.value.candidates.?[0].content.?.parts;

    const fr_response = try std.json.parseFromSlice(
        std.json.Value,
        std.heap.page_allocator,
        weather_result,
        .{},
    );
    defer fr_response.deinit();

    const fn_response_parts = [_]zenai.gemini.types.Part{.{
        .functionResponse = .{
            .name = fc.name,
            .response = fr_response.value,
        },
    }};

    const history = [_]zenai.gemini.types.Content{
        .{ .role = "user", .parts = &user_parts },
        .{ .role = "model", .parts = model_parts },
        .{ .role = "user", .parts = &fn_response_parts },
    };

    var response2 = try client.generateContent(
        model,
        &history,
        .{ .temperature = 0 },
        request_options,
    );
    defer response2.deinit();

    // Step 5: Print the final response
    if (response2.value.text()) |text| {
        std.debug.print("Model: {s}\n", .{text});
    }
}
