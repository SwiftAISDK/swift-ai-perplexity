@testable import SwiftAIPerplexity
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Suite("WeeklyPerplexity20261023")
struct WeeklyPerplexity20261023Tests {
    private var questions: [String: AIDecisionQuestion] { [
        "choice": .choice(criteria: ["a": "A", "b": "B"]),
        "score": .score(criteria: ["low", "high"]),
        "flag": .boolean(instructions: "True?")
    ] }
    private var response: AIHTTPResponse { AIHTTPResponse(statusCode: 200, headers: ["x-request-id": "req"], body: Data(#"{"model":"resolved","answers":{"choice":{"type":"choice","choice":"a","probabilities":{"a":0.75,"b":0.25},"confidence":0.9},"score":{"type":"score","score":0.8,"probabilities":{"0":0.2,"1":0.8},"legend":{"0":"low","1":"high"}},"flag":{"type":"noul","noul":0.9}},"usage":{"input_tokens":20,"output_tokens":3}}"#.utf8)) }

    @Test func nativeDecisionsPreserveEvidenceMetadataAndWarnings() async throws {
        let transport = RecordingTransport(response: response)
        let provider = try AIProviders.perplexity(configuration: .init(apiKey: "key", transport: transport))
        #expect(provider.supportedCapabilities.contains(.decision))
        let model = try provider.decisionModel("pplx-judge")
        let result = try await model.doDecide(.init(state: [.text("Inspect"), .json(["label": "photo"]), .file(mediaType: "image/png", data: .base64("iVBORw=="))], questions: questions, maxOutputTokens: 40, reasoning: "low", headers: ["x-extra": "value"], providerOptions: ["perplexity": ["custom": true]]))
        #expect(model.providerID == "perplexity.decision")
        #expect(model.supportedQuestionTypes == [.choice, .score, .boolean])
        #expect(result.answers["flag"] == .boolean(probability: 0.9))
        #expect(result.answers["choice"] == .choice(choice: "a", probabilities: ["a": 0.75, "b": 0.25]))
        #expect(result.rounding == nil)
        #expect(result.usage == .init(inputTokens: 20, outputTokens: 3))
        #expect(result.providerMetadata["perplexity"]?["confidence"]?["choice"] == 0.9)
        #expect(result.providerMetadata["perplexity"]?["legend"]?["score"]?["1"] == "high")
        #expect(result.response?.id == "req")
        #expect(result.response?.modelID == "resolved")
        #expect(result.warnings.map(\.feature) == ["maxOutputTokens", "reasoning", "providerOptions.perplexity.custom"])
        let request = try #require(await transport.requests().first)
        #expect(request.url.absoluteString == "https://api.perplexity.ai/v1/decisions")
        #expect(request.headers["authorization"] == "Bearer key")
        #expect(request.headers["x-extra"] == "value")
        let body = try decodeJSONBody(try #require(request.body))
        #expect(body["state"] == ["Inspect", ["label": "photo"], ["type": "image_url", "image_url": ["url": "data:image/png;base64,iVBORw=="]]])
        #expect(body["questions"]?["choice"]?["instructions"] == .null)
        #expect(body["questions"]?["flag"]?["type"] == "noul")
        #expect(body["maxOutputTokens"] == nil && body["reasoning"] == nil)
    }

    @Test(arguments: [JSONValue.string("state"), ["state": true], [1, 2], .number(1), .null])
    func singleJSONStatePreservesSupportedShape(value: JSONValue) async throws {
        let transport = RecordingTransport(response: response)
        let model = try AIProviders.perplexity(configuration: .init(apiKey: "key", transport: transport)).decisionModel("judge")
        _ = try await model.doDecide(.init(state: [.json(value)], questions: questions))
        let body = try decodeJSONBody(try #require(await transport.requests().first?.body))
        let expected: JSONValue
        switch value { case .string, .array, .object: expected = value; default: expected = .array([value]) }
        #expect(body["state"] == expected)
    }

    @Test func questionLimitsAndBoxFailBeforeHTTP() async throws {
        let transport = RecordingTransport(response: response)
        let model = try AIProviders.perplexity(configuration: .init(apiKey: "key", transport: transport)).decisionModel("judge")
        let invalid: [[String: AIDecisionQuestion]] = [[:], ["": .boolean(instructions: "x")],
            ["q": .choice(criteria: [:])], ["q": .score(criteria: [])], ["q": .score(criteria: [.null])],
            ["q": .score(criteria: Array(repeating: "level", count: 11))],
            ["q": .choice(criteria: Dictionary(uniqueKeysWithValues: (0...255).map { (String($0), JSONValue.null) }))],
            Dictionary(uniqueKeysWithValues: (0...128).map { (String($0), .boolean(instructions: "x")) })]
        for value in invalid {
            await #expect(throws: AIError.self) { try await model.doDecide(.init(state: [], questions: value)) }
        }
        await #expect(throws: AIDecisionUnsupportedQuestionTypeError.self) { try await model.doDecide(.init(state: [], questions: ["q": .box(instructions: "x")])) }
        #expect(await transport.requests().isEmpty)
    }

    @Test func imageValidationRecursesAndCountsAllSharedEvidence() async throws {
        let transport = RecordingTransport(response: response)
        let model = try AIProviders.perplexity(configuration: .init(apiKey: "key", transport: transport)).decisionModel("judge")
        let image: JSONValue = ["type": "image_url", "image_url": ["url": "data:image/png;base64,AA=="]]
        await #expect(throws: AIError.self) { try await model.doDecide(.init(state: [.json(["nested": .array(Array(repeating: image, count: 9))])], questions: questions)) }
        for url in ["https://example.com/a.png", "data:image/gif;base64,AA=="] {
            await #expect(throws: AIDecisionUnsupportedFunctionalityError.self) { try await model.doDecide(.init(state: [.json(["nested": ["type": "image_url", "image_url": ["url": .string(url)]]])], questions: questions)) }
        }
        #expect(await transport.requests().isEmpty)
        _ = try await model.doDecide(.init(state: [.json(.array(Array(repeating: image, count: 8)))], questions: questions))
        #expect(await transport.requests().count == 1)
    }

    @Test(arguments: [#"{"answers":{}}"#, #"{"answers":{"q":{"type":"noul","noul":2}}}"#, #"{"answers":{"q":{"type":"choice","choice":"a","probabilities":{"a":-1}}}}"#])
    func malformedResponsesRejectInvalidDistributions(json: String) async throws {
        let model = try AIProviders.perplexity(configuration: .init(apiKey: "key", transport: RecordingTransport(response: jsonResponse(json)))).decisionModel("judge")
        await #expect(throws: AIError.self) { try await model.doDecide(.init(state: [], questions: ["q": .boolean(instructions: "x")])) }
    }

    @Test func facadeRetriesTransientFailuresAndPreservesFullPrecision() async throws {
        let transport = RecordingTransport(responses: [AIHTTPResponse(statusCode: 503, headers: ["retry-after-ms": "0"], body: Data(#"{"error":{"message":"busy","code":null}}"#.utf8)), response])
        let model = try AIProviders.perplexity(configuration: .init(apiKey: "key", transport: transport)).decisionModel("judge")
        let result = try await AI.experimentalDecide(model: model, state: "state", questions: questions, maxRetries: 1)
        #expect(result.usage.totalTokens == 23)
        #expect(result.answers["score"] == .score(score: 0.8, probabilities: ["0": 0.2, "1": 0.8]))
        #expect(await transport.requests().count == 2)
    }
}
