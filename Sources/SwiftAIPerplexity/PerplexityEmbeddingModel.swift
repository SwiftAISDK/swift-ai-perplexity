import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class PerplexityEmbeddingModel: EmbeddingModel, @unchecked Sendable {
    public let providerID = "perplexity.embedding"
    public let modelID: String
    public var maxEmbeddingsPerCall: Int? { 512 }
    private let config: PerplexityModelConfiguration

    public init(modelID: String, config: PerplexityModelConfiguration) {
        self.modelID = modelID
        self.config = config
    }

    public func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResult {
        let embeddingLimit = maxEmbeddingsPerCall ?? 512
        guard request.values.count <= embeddingLimit else {
            throw AITooManyEmbeddingValuesForCallError(
                provider: providerID,
                modelID: modelID,
                maxEmbeddingsPerCall: embeddingLimit,
                values: request.values
            )
        }

        let options = try perplexityEmbeddingOptions(from: request)
        let encodingFormat = options.encodingFormat
        var body: [String: JSONValue] = [
            "model": .string(modelID),
            "input": .array(request.values),
            "encoding_format": .string(encodingFormat)
        ]
        if let dimensions = options.dimensions {
            body["dimensions"] = .number(Double(dimensions))
        }

        let response = try await config.sendJSONResponse(
            path: "/v1/embeddings",
            modelID: modelID,
            body: .object(body),
            headers: request.headers,
            abortSignal: request.abortSignal
        )
        let raw = response.json
        guard let data = raw["data"]?.arrayValue else {
            throw AIError.invalidResponse(provider: providerID, message: "No embedding data found.")
        }
        let embeddings = try data.map { item -> [Double] in
            guard let value = item["embedding"]?.stringValue,
                  let bytes = Data(base64Encoded: value) else {
                throw AIError.invalidResponse(provider: providerID, message: "Perplexity returned an invalid base64 embedding.")
            }
            switch encodingFormat {
            case "base64_binary":
                return bytes.map { Double($0) }
            default:
                return bytes.map { Double(Int8(bitPattern: $0)) }
            }
        }

        let usageValue = raw["usage"]
        let promptTokens = usageValue?["prompt_tokens"]?.intValue
        let cost = usageValue?["cost"]
        let providerMetadata: [String: JSONValue]
        if let cost, cost != .null {
            providerMetadata = [
                "perplexity": .object([
                    "cost": .object([
                        "inputCost": cost["input_cost"] ?? .null,
                        "totalCost": cost["total_cost"] ?? .null,
                        "currency": cost["currency"] ?? .null
                    ])
                ])
            ]
        } else {
            providerMetadata = [:]
        }

        return EmbeddingResult(
            embeddings: embeddings,
            usage: promptTokens.map { TokenUsage(inputTokens: $0, totalTokens: $0, rawValue: usageValue) },
            rawValue: raw,
            providerMetadata: providerMetadata,
            requestMetadata: AIRequestMetadata(body: .object(body), headers: request.headers),
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }
}

private struct PerplexityEmbeddingOptions {
    var dimensions: Int?
    var encodingFormat: String
}

private func perplexityEmbeddingOptions(from request: EmbeddingRequest) throws -> PerplexityEmbeddingOptions {
    var values: [String: JSONValue] = [:]
    if let nested = request.extraBody["perplexity"]?.objectValue {
        values.merge(nested) { _, new in new }
    }
    if let nested = request.providerOptions["perplexity"] {
        guard nested != .null else {
            return PerplexityEmbeddingOptions(dimensions: request.dimensions, encodingFormat: "base64_int8")
        }
        guard let object = nested.objectValue else {
            throw AIError.invalidArgument(argument: "providerOptions.perplexity", message: "Perplexity provider options must be an object.")
        }
        values.merge(object) { _, new in new }
    }

    let dimensions: Int?
    if let value = values["dimensions"], value != .null {
        guard let parsed = value.intValue, value.doubleValue == Double(parsed), parsed > 0 else {
            throw AIError.invalidArgument(argument: "providerOptions.perplexity.dimensions", message: "Perplexity dimensions must be a positive integer.")
        }
        dimensions = parsed
    } else {
        dimensions = request.dimensions
    }

    let encodingFormat: String
    if let value = values["encodingFormat"], value != .null {
        guard let parsed = value.stringValue, ["base64_int8", "base64_binary"].contains(parsed) else {
            throw AIError.invalidArgument(argument: "providerOptions.perplexity.encodingFormat", message: "Perplexity encodingFormat must be base64_int8 or base64_binary.")
        }
        encodingFormat = parsed
    } else {
        encodingFormat = "base64_int8"
    }
    return PerplexityEmbeddingOptions(dimensions: dimensions, encodingFormat: encodingFormat)
}