import Foundation
import Testing
@testable import SwiftAIPerplexity
import SwiftAIProvider
import SwiftAITransport

@Suite("Weekly20261114 Perplexity grounding")
struct Weekly20261114GroundingTests {
    @Test func webGroundingPreservesEvidenceExactToolCountsAndZeroStaleness() async throws {
        let raw: JSONValue = ["answers": ["q": ["type": "noul", "noul": 0.9]],
            "search_results": [["title": "Webb", "url": "https://example.com/a", "snippet": "December", "date": "2021-12-25", "last_updated": "2026-11-01"]],
            "fetched_results": [["title": "Page", "url": "https://example.com/b", "snippet": "Body"]],
            "usage": ["tool_calls_details": ["search_web": ["invocation": 2], "fetch_url": ["invocation": 0], "future": [:]]]]
        let transport = RecordingTransport(response: .init(statusCode: 200, body: try encodeJSONBody(raw)))
        let provider = try PerplexityProvider(configuration: .init(apiKey: "key", transport: transport))
        let result = try await provider.decisionModel("pplx-decider-v1.2").doDecide(.init(state: [.text("Question")], questions: ["q": .boolean()],
            providerOptions: PerplexityDecisionModelOptions(mode: .webGrounding, maxStaleness: 0).providerOptions))
        #expect(result.warnings.isEmpty)
        #expect(result.providerMetadata["perplexity"]?["searchResults"] == raw["search_results"])
        #expect(result.providerMetadata["perplexity"]?["fetchedResults"] == raw["fetched_results"])
        #expect(result.providerMetadata["perplexity"]?["toolCalls"]?["fetch_url"]?["invocation"] == 0)
        #expect(result.providerMetadata["perplexity"]?["toolCalls"]?["future"]?["invocation"] == .null)
        let body = try decodeJSONBody(#require(await transport.requests().first?.body))
        #expect(body["mode"] == "web_grounding"); #expect(body["max_staleness"] == 0)
    }
    @Test(arguments: [JSONValue.object(["mode": "invalid"]), ["maxStaleness": -1], ["maxStaleness": 0.5], ["maxStaleness": 1e20]])
    func invalidGroundingDoesNotSend(_ options: JSONValue) async throws {
        let transport = RecordingTransport(response: jsonResponse("{}"))
        let model = try PerplexityProvider(configuration: .init(apiKey: "key", transport: transport)).decisionModel("m")
        await #expect(throws: AIError.self) { try await model.doDecide(.init(state: [], questions: ["q": .boolean()], providerOptions: ["perplexity": options])) }
        #expect(await transport.requests().isEmpty)
    }
    @Test func groundingErrorHeadersRetainCompletedCounts() async throws {
        let transport = RecordingTransport(response: .init(statusCode: 400, headers: ["x-perplexity-search-queries": "0", "x-perplexity-fetched-urls": "1"], body: Data(#"{"error":{"message":"Limit"}}"#.utf8)))
        let model = try PerplexityProvider(configuration: .init(apiKey: "key", transport: transport)).decisionModel("m")
        do { _ = try await model.doDecide(.init(state: [], questions: ["q": .boolean()])); Issue.record("Expected HTTP error") }
        catch AIError.apiCall(let error) { #expect(error.responseHeaders["x-perplexity-search-queries"] == "0"); #expect(error.responseHeaders["x-perplexity-fetched-urls"] == "1"); #expect(!error.isRetryable) }
    }
}
