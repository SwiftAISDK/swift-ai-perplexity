import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities

public struct PerplexityProviderSettings: Sendable {
    public var apiKey: String?
    public var baseURL: String?
    public var headers: [String: String]
    public var queryParams: [String: String]
    public var transformRequestBody: (@Sendable ([String: JSONValue]) -> [String: JSONValue])?
    public var transport: any AITransport

    public init(apiKey: String? = nil, baseURL: String? = nil, headers: [String: String] = [:],
                queryParams: [String: String] = [:], transport: any AITransport = URLSessionTransport.shared,
                transformRequestBody: (@Sendable ([String: JSONValue]) -> [String: JSONValue])? = nil) {
        self.apiKey = apiKey; self.baseURL = baseURL; self.headers = headers
        self.queryParams = queryParams; self.transport = transport; self.transformRequestBody = transformRequestBody
    }
}

public final class PerplexityProvider: AIProvider, AIDecisionProvider, Sendable {
    public let providerID = "perplexity"
    public let supportedCapabilities: Set<ModelCapability> = [.language, .embedding, .decision]
    private let config: PerplexityModelConfiguration

    public init(configuration: PerplexityProviderSettings = .init()) throws {
        let names = ["PERPLEXITY_API_KEY"]
        guard let key = configuration.apiKey ?? environmentValue(names) else {
            throw AIError.missingAPIKey(provider: "perplexity", environmentVariables: names)
        }
        var headers = configuration.headers
        if !headers.keys.contains(where: { $0.lowercased() == "x-pplx-integration" }) { headers["X-Pplx-Integration"] = "vercel-ai-sdk" }
        headers["Authorization"] = headers["Authorization"] ?? "Bearer \(key)"
        headers = withUserAgentSuffix(headers, "ai-sdk-perplexity/5.0.20")
        let baseURL = withoutTrailingSlash(configuration.baseURL ?? "https://api.perplexity.ai")
        let http = AIProviderHTTPClient(providerID: "perplexity", headers: headers, transport: configuration.transport,
            url: { _, path in try providerURL("\(baseURL)\(path)", queryParams: configuration.queryParams) })
        config = PerplexityModelConfiguration(http: http, transformRequestBody: configuration.transformRequestBody)
    }

    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        PerplexityLanguageModel(modelID: modelID, config: config)
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        PerplexityEmbeddingModel(modelID: modelID, config: config)
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .image, modelID: modelID)
    }

    public func transcriptionModel(_ modelID: String) throws -> any TranscriptionModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .transcription, modelID: modelID)
    }

    public func speechModel(_ modelID: String) throws -> any SpeechModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .speech, modelID: modelID)
    }

    public func videoModel(_ modelID: String) throws -> any VideoModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .video, modelID: modelID)
    }

    public func rerankingModel(_ modelID: String) throws -> any RerankingModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .reranking, modelID: modelID)
    }

    public func chatModel(_ modelID: String) throws -> any LanguageModel { PerplexitySonarLanguageModel(modelID: modelID, config: config) }
    public func chat(_ modelID: String) throws -> any LanguageModel { try chatModel(modelID) }
    public func sonarModel(_ modelID: String) throws -> any LanguageModel { try chatModel(modelID) }
    public func decisionModel(_ modelID: String) throws -> any AIDecisionModelV4 {
        PerplexityDecisionModel(modelID: modelID, config: config.withProviderID("perplexity.decision"))
    }
    public func experimentalDecisionModel(_ modelID: String) throws -> any AIDecisionModelV4 { try decisionModel(modelID) }
    public func reranking(_ modelID: String) throws -> any RerankingModel { try rerankingModel(modelID) }
}

public extension AIProviders {
    static func perplexity(configuration: PerplexityProviderSettings) throws -> PerplexityProvider {
        try PerplexityProvider(configuration: configuration)
    }
}

public struct PerplexityModelConfiguration: AIProviderHTTPConfiguration {
    public var http: AIProviderHTTPClient
    public var transformRequestBody: (@Sendable ([String: JSONValue]) -> [String: JSONValue])?
    public init(http: AIProviderHTTPClient,
                transformRequestBody: (@Sendable ([String: JSONValue]) -> [String: JSONValue])? = nil) {
        self.http = http; self.transformRequestBody = transformRequestBody
    }
}
