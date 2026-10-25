@testable import SwiftAIPerplexity
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities
import Testing
import SwiftAI


@Test func perplexityEmbeddingDecodesSignedInt8AndMapsMetadata() async throws {
    let transport = RecordingTransport(response: jsonResponse(
        #"{"data":[{"embedding":"/wB/gA=="}],"usage":{"prompt_tokens":4,"cost":{"input_cost":0.01,"total_cost":0.02,"currency":"USD"}}}"#,
        headers: ["x-request-id": "pplx-embed-1"]
    ))
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(apiKey: "pplx-key", transport: transport))
    let model = try provider.embedding("pplx-embed-v1-0.6b")

    let result = try await model.embed(EmbeddingRequest(
        values: ["hello"],
        providerOptions: ["perplexity": ["dimensions": 256]]
    ))

    #expect(model.providerID == "perplexity.embedding")
    #expect(result.embeddings == [[-1, 0, 127, -128]])
    #expect(result.usage?.inputTokens == 4)
    #expect(result.providerMetadata["perplexity"]?["cost"]?["inputCost"]?.doubleValue == 0.01)
    #expect(result.providerMetadata["perplexity"]?["cost"]?["totalCost"]?.doubleValue == 0.02)
    #expect(result.providerMetadata["perplexity"]?["cost"]?["currency"]?.stringValue == "USD")
    #expect(result.responseMetadata.headers["x-request-id"] == "pplx-embed-1")

    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://api.perplexity.ai/v1/embeddings")
    #expect(request.headers["authorization"] == "Bearer pplx-key")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["model"]?.stringValue == "pplx-embed-v1-0.6b")
    #expect(body["input"]?[0]?.stringValue == "hello")
    #expect(body["dimensions"]?.intValue == 256)
    #expect(body["encoding_format"]?.stringValue == "base64_int8")
}

@Test func perplexityEmbeddingDecodesBinaryAndValidatesLimits() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"data":[{"embedding":"/wB/gA=="}]}"#))
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(apiKey: "pplx-key", transport: transport))
    let model = try provider.embeddingModel("pplx-embed-v1-4b")
    #expect(model.maxEmbeddingsPerCall == 512)

    let result = try await model.embed(EmbeddingRequest(
        values: ["hello"],
        providerOptions: ["perplexity": ["encodingFormat": "base64_binary"]]
    ))
    #expect(result.embeddings == [[255, 0, 127, 128]])

    await #expect(throws: AITooManyEmbeddingValuesForCallError.self) {
        _ = try await model.embed(EmbeddingRequest(values: Array(repeating: "x", count: 513)))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.perplexity.dimensions", message: "Perplexity dimensions must be a positive integer.")) {
        _ = try await model.embed(EmbeddingRequest(values: ["x"], providerOptions: ["perplexity": ["dimensions": 0]]))
    }
    await #expect(throws: AIError.invalidArgument(argument: "providerOptions.perplexity.encodingFormat", message: "Perplexity encodingFormat must be base64_int8 or base64_binary.")) {
        _ = try await model.embed(EmbeddingRequest(values: ["x"], providerOptions: ["perplexity": ["encodingFormat": "float"]]))
    }
}