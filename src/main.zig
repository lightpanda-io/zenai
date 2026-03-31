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

    var client = zenai.Client.init(allocator, api_key, .{});
    defer client.deinit();

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

        if (response.text()) |text| {
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
        &printStreamChunk,
    );
    std.debug.print("\n\n", .{});

    // --- Function calling ---
    std.debug.print("=== Function calling ===\n", .{});
    try functionCallingExample(&client);
}

fn printStreamChunk(response: zenai.types.GenerateContentResponse) void {
    if (response.text()) |t| {
        const fd = std.posix.STDOUT_FILENO;
        _ = std.posix.write(fd, t) catch return;
    }
}

fn functionCallingExample(client: *zenai.Client) !void {
    const model = "gemini-2.5-flash";

    // Define the tool
    const tools = [_]zenai.types.Tool{.{
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

    const request_options = zenai.Client.RequestOptions{
        .tools = &tools,
    };

    // Step 1: Send the user prompt
    std.debug.print("User: What's the weather like in Paris?\n", .{});

    const user_parts = [_]zenai.types.Part{.{ .text = "What's the weather like in Paris?" }};
    const user_content = [_]zenai.types.Content{.{ .role = "user", .parts = &user_parts }};

    var response1 = try client.generateContent(
        model,
        &user_content,
        .{ .temperature = 0 },
        request_options,
    );
    defer response1.deinit();

    // Step 2: Check if the model wants to call a function
    const fc = response1.firstFunctionCall() orelse {
        std.debug.print("Model responded with text: {s}\n", .{response1.text() orelse "no text"});
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

    const fn_response_parts = [_]zenai.types.Part{.{
        .functionResponse = .{
            .name = fc.name,
            .response = fr_response.value,
        },
    }};

    const history = [_]zenai.types.Content{
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
    if (response2.text()) |text| {
        std.debug.print("Model: {s}\n", .{text});
    }
}
