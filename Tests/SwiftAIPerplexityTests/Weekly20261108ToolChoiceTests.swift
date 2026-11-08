import Foundation
import Testing
@testable import SwiftAIPerplexity
import SwiftAIProvider

@Suite("Weekly20261108 Perplexity tool disable")
struct Weekly20261108ToolChoiceTests {
    @Test(arguments: ["low", "openai/gpt-5.1"], [JSONValue.string("none"), ["type": "none"]])
    func noneDisablesPresetAndExplicitTools(_ modelID: String, _ choice: JSONValue) async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"id":"r","object":"response","created_at":1,"status":"completed","model":"model","output":[]}"#))
        let provider = try PerplexityProvider(configuration: .init(apiKey: "key", transport: transport))
        let result = try await provider.languageModel(modelID).generate(.init(messages: [
            .user("Hello"),
            .init(role: .assistant, content: [.toolCall(.init(id: "call", name: "lookup", arguments: "{}"))]),
            .init(role: .tool, content: [.toolResult(.init(toolCallID: "call", toolName: "lookup", result: "done"))])
        ], tools: ["ignored": ["type": "provider", "id": "perplexity.web_search"]], toolChoice: choice, providerOptions: ["perplexity": ["tools": [["type": "mcp", "server_label": "test", "server_url": "https://example.com/mcp", "authorization": "secret"]], "max_tool_calls": 5]]))
        let request = try #require(await transport.requests().first)
        let bytes = try #require(request.body)
        let body = try JSONDecoder().decode(JSONValue.self, from: bytes)
        #expect(body["tools"] == nil)
        #expect(body["max_tool_calls"] == 0)
        #expect(body["input"]?.arrayValue?.contains { $0["type"] == "function_call" && $0["call_id"] == "call" } == true)
        #expect(body["input"]?.arrayValue?.contains { $0["type"] == "function_call_output" && $0["call_id"] == "call" } == true)
        #expect(!String(decoding: bytes, as: UTF8.self).contains("secret"))
        #expect(result.warnings.isEmpty)
    }
}
