import Foundation
import Testing
import SwiftAIProvider
import SwiftAITransport
@testable import SwiftAIPerplexity

@Suite struct Weekly20261101PerplexityBehaviorTests {
    private func response(_ output: [JSONValue]) -> JSONValue {
        ["id": "r", "object": "response", "created_at": 1, "status": "completed", "model": "model", "output": .array(output)]
    }

    @Test func unaryQueriesAreStableDeduplicatedAndNullWhenAbsent() async throws {
        for (output, expected) in [([JSONValue.object(["type": "search_results", "queries": ["one", "two"], "results": []]), ["type": "search_results", "queries": ["two", "three"], "results": []]], JSONValue.array(["one", "two", "three"])), ([], .null)] {
            let transport = RecordingTransport(response: jsonResponse(String(decoding: try encodeJSONBody(response(output)), as: UTF8.self)))
            let model = try PerplexityProvider(configuration: .init(apiKey: "key", transport: transport)).languageModel("model")
            let result = try await model.generate(.init(messages: [.user("Hi")], reasoning: "none"))
            #expect(result.providerMetadata["perplexity"]?["searchQueries"] == expected)
            let body = try decodeJSONBody(try #require(await transport.requests().first?.body))
            #expect(body["reasoning"]?["effort"] == "minimal")
            #expect(result.warnings.contains { $0.type == "compatibility" && $0.feature == "reasoning" })
        }
    }

    @Test func streamRetainsQueriesAcrossReasoningOutputAndCompletion() async throws {
        let events: [JSONValue] = [
            ["type": "response.reasoning.started", "sequence_number": 0, "thought": "Plan"],
            ["type": "response.reasoning.search_queries", "sequence_number": 1, "queries": ["one", "two"]],
            ["type": "response.reasoning.search_queries", "sequence_number": 2, "queries": ["two"]],
            ["type": "response.reasoning.fetch_url_queries", "sequence_number": 3, "urls": ["https://example.com"]],
            ["type": "response.output_item.done", "sequence_number": 4, "item": ["type": "search_results", "queries": ["three", "one"], "results": []]],
            ["type": "response.completed", "sequence_number": 5, "response": response([])]
        ]
        let sse = try events.map { "data: " + String(decoding: try encodeJSONBody($0), as: UTF8.self) }.joined(separator: "\n\n")
        let transport = RecordingTransport(response: sseResponse(sse))
        let model = try PerplexityProvider(configuration: .init(apiKey: "key", transport: transport)).languageModel("model")
        var terminal: [String: JSONValue] = [:]
        for try await part in model.stream(.init(messages: [.user("Hi")])) {
            if case let .finishMetadata(_, _, metadata) = part { terminal = metadata }
        }
        #expect(terminal["perplexity"]?["searchQueries"] == ["one", "two", "three"])
    }

    @Test func malformedQueryArraysFailResponseValidation() async throws {
        let value = response([["type": "search_results", "queries": ["valid", 1], "results": []]])
        let transport = RecordingTransport(response: jsonResponse(String(decoding: try encodeJSONBody(value), as: UTF8.self)))
        let model = try PerplexityProvider(configuration: .init(apiKey: "key", transport: transport)).languageModel("model")
        await #expect(throws: AIError.self) { try await model.generate(.init(messages: [.user("Hi")])) }
    }
}
