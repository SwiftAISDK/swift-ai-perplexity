@testable import SwiftAIPerplexity
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities
import Testing
import SwiftAI

@Test func perplexityLanguageModelUsesAgentAPIByDefault() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"resp-agent","created_at":1710000000,"model":"pplx","object":"response","status":"completed","output":[{"type":"message","id":"msg-1","content":[{"type":"output_text","text":"agent answer","annotations":[]}]}]}
    """))
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(apiKey: "pplx-key", transport: transport))
    let result = try await provider.languageModel("fast").generate(.init(messages: [.user("Research")]))

    #expect(result.text == "agent answer")
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://api.perplexity.ai/v1/agent")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["preset"]?.stringValue == "fast")
    #expect(body["model"] == nil)
}

@Test func perplexityChatModelUsesNativeSonarShapeAndKeepsMetadata() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"ppl-1","created":1710000000,"model":"sonar","choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"citations":["https://example.com/a"],"images":[{"image_url":"https://img.example.com/a.png","origin_url":"https://origin.example.com","height":512,"width":768}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7,"reasoning_tokens":1,"citation_tokens":2,"num_search_queries":1,"cost":{"request_cost":0.01,"total_cost":0.02}}}
    """))
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(apiKey: "pplx-key", transport: transport))
    let model = try provider.chatModel("sonar")

    let result = try await model.generate(LanguageModelRequest(
        messages: [
            .system("Search carefully."),
            AIMessage(role: .user, content: [
                .text("Look at this."),
                .imageURL("https://example.com/image.png"),
                .file(mimeType: "application/pdf", data: Data("pdf".utf8), filename: "brief.pdf")
            ])
        ],
        temperature: 0.2,
        topP: 0.9,
        maxOutputTokens: 64,
        stopSequences: ["ignored"],
        extraBody: ["search_mode": "academic"]
    ))

    #expect(result.text == "answer")
    #expect(result.usage?.totalTokens == 8)
    #expect(result.usage?.outputTokens == 5)
    #expect(result.usage?.inputTokensNoCache == 3)
    #expect(result.usage?.outputReasoningTokens == 1)
    #expect(result.usage?.outputTextTokens == 4)
    #expect(result.usage?.rawValue?["reasoning_tokens"]?.intValue == 1)
    #expect(result.sources.count == 1)
    let source = try #require(result.sources.first)
    #expect(source.id == "citation-0")
    #expect(source.sourceType == "url")
    #expect(source.url == "https://example.com/a")
    #expect(source.providerMetadata["perplexity"]?["citationIndex"]?.intValue == 0)
    #expect(result.rawValue["citations"]?[0]?.stringValue == "https://example.com/a")
    #expect(result.rawValue["images"]?[0]?["image_url"]?.stringValue == "https://img.example.com/a.png")
    #expect(result.providerMetadata["perplexity"]?["images"]?[0]?["imageUrl"]?.stringValue == "https://img.example.com/a.png")
    #expect(result.providerMetadata["perplexity"]?["usage"]?["citationTokens"]?.intValue == 2)
    #expect(result.providerMetadata["perplexity"]?["usage"]?["numSearchQueries"]?.intValue == 1)
    #expect(result.providerMetadata["perplexity"]?["cost"]?["requestCost"]?.doubleValue == 0.01)
    #expect(result.providerMetadata["perplexity"]?["cost"]?["totalCost"]?.doubleValue == 0.02)
    #expect(result.requestMetadata.body?["model"]?.stringValue == "sonar")
    #expect(result.requestMetadata.body?["messages"]?[1]?["content"]?[2]?["file_name"]?.stringValue == "brief.pdf")
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://api.perplexity.ai/chat/completions")
    #expect(request.headers["authorization"] == "Bearer pplx-key")
    #expect(request.headers["user-agent"] == "ai-sdk-perplexity/5.0.20")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["model"]?.stringValue == "sonar")
    #expect(body["temperature"]?.doubleValue == 0.2)
    #expect(body["top_p"]?.doubleValue == 0.9)
    #expect(body["max_tokens"]?.intValue == 64)
    #expect(body["stop"] == nil)
    #expect(body["search_mode"]?.stringValue == "academic")
    #expect(body["messages"]?[0]?["content"]?.stringValue == "Search carefully.")
    let content = body["messages"]?[1]?["content"]
    #expect(content?[0]?["type"]?.stringValue == "text")
    #expect(content?[1]?["type"]?.stringValue == "image_url")
    #expect(content?[1]?["image_url"]?["url"]?.stringValue == "https://example.com/image.png")
    #expect(content?[2]?["type"]?.stringValue == "file_url")
    #expect(content?[2]?["file_url"]?["url"]?.stringValue == Data("pdf".utf8).base64EncodedString())
    #expect(content?[2]?["file_name"]?.stringValue == "brief.pdf")
}
@Test func perplexitySonarTreatsProviderDefaultReasoningAsNoopLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"ppl-provider-default","created":1710000000,"model":"sonar","choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}
    """))
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(apiKey: "pplx-key", transport: transport))
    let model = try provider.sonarModel("sonar")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Hi")],
        reasoning: "provider-default"
    ))

    #expect(result.text == "answer")
    #expect(result.warnings == [])
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["reasoning"] == nil)
}
@Test func perplexitySonarAppliesTransformRequestBodyToGenerateAndStream() async throws {
    let generateTransport = RecordingTransport(response: jsonResponse("""
    {"id":"ppl-transform","created":1710000000,"model":"sonar","choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}
    """))
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(
        apiKey: "pplx-key",
        transport: generateTransport,
        transformRequestBody: { body in
            var body = body
            body["search_mode"] = .string("academic")
            return body
        }
    ))
    let model = try provider.sonarModel("sonar")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))

    #expect(result.text == "answer")
    #expect(result.requestMetadata.body?["search_mode"]?.stringValue == "academic")
    let generateBody = try decodeJSONBody(try #require((await generateTransport.requests()).first?.body))
    #expect(generateBody["search_mode"]?.stringValue == "academic")

    let streamTransport = RecordingTransport(response: sseResponse("""
    data: {"id":"ppl-transform","created":1710000000,"model":"sonar","choices":[{"delta":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}

    data: [DONE]

    """))
    let streamProvider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(
        apiKey: "pplx-key",
        transport: streamTransport,
        transformRequestBody: { body in
            var body = body
            body["search_domain_filter"] = .array([.string("example.com")])
            return body
        }
    ))
    let streamModel = try streamProvider.sonarModel("sonar")

    for try await _ in streamModel.stream(LanguageModelRequest(messages: [.user("Hi")])) {}

    let streamBody = try decodeJSONBody(try #require((await streamTransport.requests()).first?.body))
    #expect(streamBody["stream"]?.boolValue == true)
    #expect(streamBody["search_domain_filter"]?[0]?.stringValue == "example.com")
}
@Test func perplexitySonarMapsStructuredFormatWarningsAndMetadata() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"id":"ppl-structured","created":1710000000,"model":"sonar","choices":[{"message":{"role":"assistant","content":"{\\"ok\\":true}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}
    """, headers: ["pplx-header": "structured"]))
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(apiKey: "pplx-key", transport: transport))
    let model = try provider.sonarModel("sonar")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("JSON")],
        topK: 3,
        presencePenalty: 0.2,
        frequencyPenalty: 0.4,
        seed: 42,
        maxOutputTokens: 32,
        stopSequences: ["###"],
        responseFormat: .json(schema: [
            "type": "object",
            "properties": ["ok": ["type": "boolean"]],
            "required": ["ok"]
        ]),
        reasoning: "custom",
        providerOptions: [
            "perplexity": [
                "search_recency_filter": "month"
            ]
        ],
        extraBody: [
            "perplexity": [
                "return_images": true
            ],
            "responseFormat": [
                "type": "json",
                "schema": ["type": "string"]
            ]
        ]
    ))

    #expect(result.text == "{\"ok\":true}")
    #expect(result.warnings == [
        AIWarning(type: "unsupported", feature: "topK"),
        AIWarning(type: "unsupported", feature: "stopSequences"),
        AIWarning(type: "unsupported", feature: "seed"),
        AIWarning(type: "unsupported", feature: "reasoning", message: "This provider does not support reasoning configuration.")
    ])
    #expect(result.responseMetadata.id == "ppl-structured")
    #expect(result.responseMetadata.modelID == "sonar")
    #expect(result.responseMetadata.headers["pplx-header"] == "structured")
    #expect(result.responseMetadata.body?["usage"]?["total_tokens"]?.intValue == 11)

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["response_format"]?["type"]?.stringValue == "json_schema")
    #expect(body["response_format"]?["json_schema"]?["schema"]?["type"]?.stringValue == "object")
    #expect(body["response_format"]?["json_schema"]?["schema"]?["properties"]?["ok"]?["type"]?.stringValue == "boolean")
    #expect(body["responseFormat"] == nil)
    #expect(body["top_k"]?.intValue == 3)
    #expect(body["presence_penalty"]?.doubleValue == 0.2)
    #expect(body["frequency_penalty"]?.doubleValue == 0.4)
    #expect(body["max_tokens"]?.intValue == 32)
    #expect(body["stop"] == nil)
    #expect(body["seed"] == nil)
    #expect(body["reasoning"] == nil)
    #expect(body["return_images"]?.boolValue == true)
    #expect(body["search_recency_filter"]?.stringValue == "month")
    #expect(body["perplexity"] == nil)

    await #expect(throws: AIError.invalidArgument(argument: "messages", message: "Perplexity does not support tool messages.")) {
        _ = try await model.generate(LanguageModelRequest(messages: [
            .toolResponses(toolResults: [
                AIToolResult(toolCallID: "call-1", toolName: "lookup", result: "done")
            ])
        ]))
    }
}
@Test func perplexitySonarMapsMissingFinishReasonToOtherLikeUpstream() async throws {
    let generateTransport = RecordingTransport(response: jsonResponse("""
    {"id":"ppl-no-finish","created":1710000000,"model":"sonar","choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":null}]}
    """))
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(apiKey: "pplx-key", transport: generateTransport))
    let model = try provider.sonarModel("sonar")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))

    #expect(result.finishReason == "other")

    let streamTransport = RecordingTransport(response: sseResponse("""
    data: {"id":"ppl-no-finish","created":1710000000,"model":"sonar","choices":[{"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}

    data: [DONE]

    """))
    let streamProvider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(apiKey: "pplx-key", transport: streamTransport))
    let streamModel = try streamProvider.sonarModel("sonar")

    var finishReason: String?
    for try await part in streamModel.stream(LanguageModelRequest(messages: [.user("Hi")])) {
        if case let .finishMetadata(reason, _, _) = part {
            finishReason = reason
        }
    }

    #expect(finishReason == "other")
}
@Test func perplexitySonarGenerateResponseValidationMatchesUpstreamSchema() async throws {
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(
        apiKey: "pplx-key",
        transport: RecordingTransport(response: jsonResponse("""
        {"id":"ppl-invalid","model":"sonar","choices":[{"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}
        """))
    ))
    let model = try provider.sonarModel("sonar")

    await #expect(throws: AIError.invalidResponse(provider: "perplexity", message: "Perplexity response is invalid.")) {
        _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))
    }
}
@Test func perplexitySonarStreamsNativeChunksWithUsage() async throws {
    let transport = RecordingTransport(response: sseResponse("""
    data: {"id":"ppl-1","created":1710000000,"model":"sonar","choices":[{"delta":{"role":"assistant","content":null},"finish_reason":null}]}

    data: {"id":"ppl-1","created":1710000000,"model":"sonar","choices":[{"delta":{"role":"assistant","content":"hel"},"finish_reason":null}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3},"citations":["https://example.com/a"]}

    data: {"id":"ppl-1","created":1710000000,"model":"sonar","choices":[{"delta":{"role":"assistant","content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4,"reasoning_tokens":1,"citation_tokens":1,"num_search_queries":1}}

    data: [DONE]

    """))
    let provider = try AIProviders.perplexity(configuration: PerplexityProviderSettings(apiKey: "pplx-key", transport: transport))
    let model = try provider.sonarModel("sonar")

    var streamStartWarnings: [AIWarning]?
    var textLifecycle: [String] = []
    var sources: [AISource] = []
    var finishReason: String?
    var totalTokens: Int?
    var outputReasoningTokens: Int?
    var outputTextTokens: Int?
    var providerMetadata: [String: JSONValue] = [:]
    var metadata: AIResponseMetadata?
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {
        switch part {
        case let .streamStart(warnings):
            streamStartWarnings = warnings
        case let .textStart(id, _):
            textLifecycle.append("start:\(id)")
        case let .textDeltaPart(id, delta, _):
            textLifecycle.append("delta:\(id):\(delta)")
        case let .textEnd(id, _), let .textEndWithCitations(id, _, _):
            textLifecycle.append("end:\(id)")
        case let .source(source):
            sources.append(source)
        case let .finishMetadata(reason, usage, metadata):
            finishReason = reason
            totalTokens = usage?.totalTokens
            outputReasoningTokens = usage?.outputReasoningTokens
            outputTextTokens = usage?.outputTextTokens
            providerMetadata = metadata
        case let .responseMetadata(value):
            metadata = value
        default:
            break
        }
    }

    #expect(streamStartWarnings == [])
    #expect(textLifecycle == ["start:0", "delta:0:hel", "delta:0:lo", "end:0"])
    #expect(sources.count == 1)
    let source = try #require(sources.first)
    #expect(source.id == "citation-0")
    #expect(source.url == "https://example.com/a")
    #expect(source.providerMetadata["perplexity"]?["citationIndex"]?.intValue == 0)
    #expect(finishReason == "stop")
    #expect(totalTokens == 5)
    #expect(outputReasoningTokens == 1)
    #expect(outputTextTokens == 2)
    #expect(providerMetadata["perplexity"]?["usage"]?["citationTokens"]?.intValue == 1)
    #expect(providerMetadata["perplexity"]?["usage"]?["numSearchQueries"]?.intValue == 1)
    #expect(providerMetadata["perplexity"]?["images"] == .null)
    #expect(providerMetadata["perplexity"]?["cost"] == .null)
    #expect(metadata?.id == "ppl-1")
    #expect(metadata?.modelID == "sonar")
    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["stream"] == true)
    #expect(body["messages"]?[0]?["content"]?.stringValue == "Hi")
}