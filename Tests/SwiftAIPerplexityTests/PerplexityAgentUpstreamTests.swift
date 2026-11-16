@testable import SwiftAIPerplexity
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

// Translated from packages/perplexity/src/perplexity-language-model.test.ts,
// perplexity-decision-model.test.ts and convert-perplexity-usage.test.ts
// (matching-release ref 8bc7652c).

private let usageJSON: JSONValue = [
    "input_tokens": 120,
    "input_tokens_details": ["cache_creation_input_tokens": 10, "cache_read_input_tokens": 20],
    "output_tokens": 45,
    "output_tokens_details": ["reasoning_tokens": 5],
    "total_tokens": 165,
    "tool_calls_details": ["search_web": ["invocation": 2]],
    "cost": ["currency": "USD", "input_cost": 0.001, "output_cost": 0.002, "tool_calls_cost": 0.003, "total_cost": 0.006]
]

private let searchResultsItem: JSONValue = [
    "type": "search_results",
    "queries": ["latest AI news"],
    "results": [["id": 1, "title": "Example source", "url": "https://example.com/source", "snippet": "An example search result.", "date": "2026-08-01", "source": "web"]]
]

private let messageItem: JSONValue = [
    "id": "msg-123", "type": "message", "status": "completed", "role": "assistant",
    "content": [["type": "output_text", "text": "Hello from Perplexity.", "annotations": JSONValue.array([JSONValue]())]]
]

private let financeOutput: JSONValue = [
    "type": "finance_results", "categories": ["quote"], "tickers": ["AAPL"],
    "results": [["category": "quote", "content": "AAPL: $230.00", "sources": ["https://example.com/quote"], "tickers": ["AAPL"]]]
]

private func createResponse(_ overrides: [String: JSONValue?] = [:]) -> JSONValue {
    var response: [String: JSONValue] = [
        "id": "resp-123", "created_at": 1784292159, "model": "openai/gpt-5.1", "object": "response",
        "output": [searchResultsItem, messageItem], "status": "completed", "usage": usageJSON
    ]
    for (key, value) in overrides { response[key] = value }
    return .object(response)
}

private func json(_ value: JSONValue) -> AIHTTPResponse {
    jsonResponse(String(decoding: (try? encodeJSONBody(value)) ?? Data(), as: UTF8.self))
}

private func stream(_ events: [JSONValue]) -> AIHTTPResponse {
    sseResponse((events.compactMap { try? encodeJSONBody($0) }.map { "data: \(String(decoding: $0, as: UTF8.self))\n\n" } + ["data: [DONE]\n\n"]).joined())
}

private func createStreamChunks(_ overrides: [String: JSONValue?] = [:]) -> [JSONValue] {
    [
        ["type": "response.created", "sequence_number": 0, "response": createResponse(["output": JSONValue.array([JSONValue]()), "usage": nil])],
        ["type": "response.reasoning.search_results", "sequence_number": 1, "results": [["id": 1, "title": "Example source", "url": "https://example.com/source", "snippet": "An example search result.", "source": "web"]]],
        ["type": "response.output_text.delta", "sequence_number": 2, "item_id": "msg-123", "output_index": 1, "content_index": 0, "delta": "Hello "],
        ["type": "response.output_text.delta", "sequence_number": 3, "item_id": "msg-123", "output_index": 1, "content_index": 0, "delta": "from Perplexity."],
        ["type": "response.output_text.done", "sequence_number": 4, "item_id": "msg-123", "output_index": 1, "content_index": 0, "text": "Hello from Perplexity."],
        ["type": "response.completed", "sequence_number": 5, "response": createResponse(overrides)]
    ]
}

private func makeModel(_ response: AIHTTPResponse, modelID: String = "low") throws -> (any LanguageModel, RecordingTransport) {
    let transport = RecordingTransport(response: response)
    return (try PerplexityProvider(configuration: .init(apiKey: "test-token", transport: transport)).languageModel(modelID), transport)
}

private func sentBody(_ transport: RecordingTransport) async throws -> JSONValue {
    try decodeJSONBody(try #require(await transport.requests().first?.body))
}

private func collect(_ model: any LanguageModel, _ request: LanguageModelRequest) async throws -> [LanguageStreamPart] {
    var parts: [LanguageStreamPart] = []
    for try await part in model.stream(request) { parts.append(part) }
    return parts
}

private let hello = LanguageModelRequest(messages: [.user("Hello")])

private func finishPart(_ parts: [LanguageStreamPart]) -> (reason: String?, usage: TokenUsage?, metadata: [String: JSONValue])? {
    for part in parts { if case let .finishMetadata(reason, usage, metadata) = part { return (reason, usage, metadata) } }
    return nil
}

@Suite("Perplexity upstream doGenerate")
struct PerplexityUpstreamGenerateTests {
    @Test func collectsSearchQueriesAcrossSearchResultItemsWithoutDuplicates() async throws {
        let (model, _) = try makeModel(json(createResponse(["output": [
            ["type": "search_results", "queries": ["running shoes Denmark", "best running shoe brands"], "results": JSONValue.array([JSONValue]())],
            ["type": "search_results", "queries": ["best running shoe brands", "running shoes 2026"], "results": JSONValue.array([JSONValue]())]
        ]])))
        let result = try await model.generate(hello)
        #expect(result.providerMetadata["perplexity"]?["searchQueries"] == ["running shoes Denmark", "best running shoe brands", "running shoes 2026"])
    }

    @Test func returnsNullSearchQueriesWhenNoWebSearchRan() async throws {
        let (model, _) = try makeModel(json(createResponse(["output": [messageItem]])))
        #expect(try await model.generate(hello).providerMetadata["perplexity"]?["searchQueries"] == .null)
    }

    @Test func acceptsNativeToolResultsWithoutTreatingThemAsWebSearchResults() async throws {
        let response = createResponse(["output": [financeOutput, searchResultsItem, messageItem]])
        let (model, _) = try makeModel(json(response))
        let result = try await model.generate(hello)
        #expect(result.content.contains(.text("Hello from Perplexity.")))
        #expect(result.responseMetadata.body == response)
        #expect(result.finishReason == "stop")
    }

    @Test func stillRejectsMalformedHandledOutputItems() async throws {
        let (model, _) = try makeModel(json(createResponse(["output": [["type": "search_results", "results": [["title": "Missing URL"]]]]])))
        await #expect(throws: AIError.self) { _ = try await model.generate(hello) }
    }

    @Test func extractsTextSourcesUsageCostAndResponseMetadata() async throws {
        let (model, _) = try makeModel(json(createResponse()))
        let result = try await model.generate(hello)
        #expect(result.content.count == 2)
        guard case let .source(source) = result.content.first else { Issue.record("expected source"); return }
        #expect(source.id == "1" && source.url == "https://example.com/source" && source.title == "Example source")
        #expect(result.content.last == .text("Hello from Perplexity."))
        #expect(result.usage?.inputTokens == 120 && result.usage?.inputTokensNoCache == 90 && result.usage?.inputTokensCacheRead == 20 && result.usage?.inputTokensCacheWrite == 10)
        #expect(result.usage?.outputTokens == 45 && result.usage?.outputTextTokens == 40 && result.usage?.outputReasoningTokens == 5)
        #expect(result.usage?.rawValue == usageJSON)
        #expect(result.providerMetadata == ["perplexity": [
            "usage": ["citationTokens": .null, "numSearchQueries": 2],
            "images": .null,
            "searchQueries": ["latest AI news"],
            "cost": ["inputTokensCost": 0.001, "outputTokensCost": 0.002, "requestCost": .null, "totalCost": 0.006, "currency": "USD", "cacheCreationCost": .null, "cacheReadCost": .null, "toolCallsCost": 0.003],
            "toolCalls": ["search_web": ["invocation": 2]]
        ]])
        #expect(result.responseMetadata.id == "resp-123")
        #expect(result.responseMetadata.modelID == "openai/gpt-5.1")
        #expect(result.responseMetadata.timestamp == Date(timeIntervalSince1970: 1784292159))
    }

    @Test func reportsZeroSearchCountsForEmptyToolCallDetails() async throws {
        var usage = usageJSON.objectValue ?? [:]
        usage["tool_calls_details"] = .object([:])
        let (model, _) = try makeModel(json(createResponse(["usage": .object(usage)])))
        let metadata = try await model.generate(hello).providerMetadata["perplexity"]
        #expect(metadata?["usage"]?["numSearchQueries"] == 0)
        #expect(metadata?["toolCalls"] == .object([:]))
    }

    @Test(arguments: [("low", "preset"), ("openai/gpt-5.1", "model"), ("sonar-pro", "model")])
    func sendsPresetsAndDirectModelIDs(_ modelID: String, _ key: String) async throws {
        let (model, transport) = try makeModel(json(createResponse()), modelID: modelID)
        let result = try await model.generate(hello)
        #expect(try await sentBody(transport) == [.init(stringLiteral: key): .string(modelID), "input": [["type": "message", "role": "user", "content": "Hello"]]])
        #expect(!result.warnings.contains { $0.type == "deprecated" })
    }

    @Test func passesAgentAPIProviderOptionsAndFunctionTools() async throws {
        let (model, transport) = try makeModel(json(createResponse()))
        let schema: JSONValue = ["type": "object", "properties": ["answer": ["type": "string"]], "required": ["answer"], "additionalProperties": false]
        _ = try await model.generate(LanguageModelRequest(
            messages: [.user("Hello")],
            temperature: 0.4,
            topP: 0.9,
            maxOutputTokens: 200,
            responseFormat: .json(schema: schema, name: "answer"),
            reasoning: "high",
            tools: ["weather": ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"], "description": "Get the weather", "strict": true]],
            providerOptions: ["perplexity": [
                "max_steps": 4, "previous_response_id": "resp-previous", "store": false,
                "tools": [["type": "web_search", "search_context_size": "low"]],
                "future_option": ["enabled": true]
            ]]
        ))
        #expect(try await sentBody(transport) == [
            "preset": "low",
            "input": [["type": "message", "role": "user", "content": "Hello"]],
            "max_output_tokens": 200,
            "temperature": 0.4,
            "top_p": 0.9,
            "reasoning": ["effort": "high"],
            "max_steps": 4,
            "previous_response_id": "resp-previous",
            "store": false,
            "future_option": ["enabled": true],
            "response_format": ["type": "json_schema", "json_schema": ["name": "answer", "schema": schema, "strict": true]],
            "tools": [
                ["type": "web_search", "search_context_size": "low"],
                ["type": "function", "name": "weather", "description": "Get the weather", "parameters": ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]], "strict": true]
            ]
        ])
    }

    @Test(arguments: [("max", "xhigh"), ("none", "minimal")])
    func mapsTopLevelReasoningWithAWarning(_ reasoning: String, _ effort: String) async throws {
        let (model, transport) = try makeModel(json(createResponse()))
        let result = try await model.generate(LanguageModelRequest(messages: [.user("Hello")], reasoning: reasoning))
        #expect(try await sentBody(transport)["reasoning"] == ["effort": .string(effort)])
        #expect(result.warnings.contains(AIWarning(type: "compatibility", feature: "reasoning", message: "reasoning \"\(reasoning)\" is not directly supported by this model. mapped to effort \"\(effort)\".")))
    }

    @Test func rejectsInvalidProviderOptions() async throws {
        let (model, transport) = try makeModel(json(createResponse()))
        await #expect(throws: AIError.self) { _ = try await model.generate(LanguageModelRequest(messages: [.user("Hello")], providerOptions: ["perplexity": ["max_steps": 0]])) }
        #expect(await transport.requests().isEmpty)
    }

    @Test(arguments: [true, false])
    func extractsFunctionCalls(_ withSignature: Bool) async throws {
        var item: [String: JSONValue] = ["id": "fc-123", "type": "function_call", "status": "completed", "call_id": "call-123", "name": "weather", "arguments": #"{"city":"San Francisco"}"#]
        if withSignature { item["thought_signature"] = "signature-123" }
        let (model, _) = try makeModel(json(createResponse(["status": "requires_action", "output": [.object(item)]])))
        let result = try await model.generate(hello)
        let call = try #require(result.toolCalls.first)
        #expect(call.id == "call-123" && call.name == "weather" && call.arguments == #"{"city":"San Francisco"}"#)
        #expect(call.providerMetadata == ["perplexity": withSignature ? ["itemId": "fc-123", "thoughtSignature": "signature-123"] : ["itemId": "fc-123"]])
        #expect(result.finishReason == "tool-calls")
    }

    @Test(arguments: [("max_output_tokens", "length"), ("content_filter", "content-filter")])
    func mapsIncompleteReasons(_ reason: String, _ expected: String) async throws {
        let (model, _) = try makeModel(json(createResponse(["status": "incomplete", "incomplete_details": ["reason": .string(reason)]])))
        #expect(try await model.generate(hello).finishReason == expected)
    }

    @Test func throwsAnAPICallErrorForFailedResponsesReturnedWithHTTP200() async throws {
        let (model, _) = try makeModel(json(createResponse(["status": "failed", "error": ["message": "Agent run failed", "type": "server_error"], "output": JSONValue.array([JSONValue]())])))
        do {
            _ = try await model.generate(hello)
            Issue.record("expected failure")
        } catch AIError.apiCall(let error) {
            #expect(error.message == "Agent run failed")
            #expect(error.statusCode == 400)
            #expect(!error.isRetryable)
        }
    }

    @Test func rejectsMalformedSuccessfulResponses() async throws {
        let (model, _) = try makeModel(json(["id": "resp-123", "created_at": 1784292159, "model": "openai/gpt-5.1", "object": "response", "status": "completed"]))
        await #expect(throws: AIError.self) { _ = try await model.generate(hello) }
    }

    @Test func keepsTheSearchResultIDWhenAnAnnotationWithTheSameURLAppearsFirst() async throws {
        let (model, _) = try makeModel(json(createResponse(["output": [
            ["id": "msg-123", "type": "message", "role": "assistant", "content": [["type": "output_text", "text": "Answer", "annotations": [["type": "url_citation", "url": "https://example.com/source", "title": "Annotation title"]]]]],
            ["type": "search_results", "results": [["id": 7, "title": "Search result title", "url": "https://example.com/source", "snippet": "Search result snippet."]]]
        ]])))
        let source = try await model.generate(hello).content.compactMap { part -> AISource? in if case let .source(source) = part { source } else { nil } }.first
        #expect(source?.id == "7" && source?.title == "Search result title")
        #expect(source?.providerMetadata["perplexity"]?["resultId"] == 7)
    }

    @Test func mapsHTTPErrorEnvelopesToTheirMessage() async throws {
        for (body, message) in [
            (#"{"error":{"message":"Invalid model","type":"invalid_request_error","code":400}}"#, "Invalid model"),
            (#"{"error":"Plain error"}"#, "Plain error"),
            (#"{"detail":[{"msg":"first"},{"msg":"second"}]}"#, "first, second"),
            (#"{"detail":"Not found"}"#, "Not found")
        ] {
            let (model, _) = try makeModel(AIHTTPResponse(statusCode: 400, headers: [:], body: Data(body.utf8)))
            do {
                _ = try await model.generate(hello)
                Issue.record("expected failure")
            } catch AIError.apiCall(let error) {
                #expect(error.message == message)
                #expect(error.responseBody == body)
            }
        }
    }
}

@Suite("Perplexity upstream doStream")
struct PerplexityUpstreamStreamTests {
    @Test(arguments: ["response.completed", "response.incomplete"])
    func preservesAnExplicitTerminalEventWithoutResponseDetails(_ type: String) async throws {
        let (model, _) = try makeModel(stream([["type": .string(type)]]))
        let parts = try await collect(model, hello)
        #expect(!parts.contains { if case .error = $0 { true } else { false } })
        #expect(finishPart(parts)?.reason == "other")
    }

    @Test func collectsSearchQueriesFromReasoningEventsWhenTheCompletedResponseOmitsThem() async throws {
        let (model, _) = try makeModel(stream([
            ["type": "response.reasoning.started", "sequence_number": 0, "thought": "Planning..."],
            ["type": "response.reasoning.search_queries", "sequence_number": 1, "queries": ["running shoes Denmark", "best running shoe brands"], "thought": "Searching the web..."],
            ["type": "response.reasoning.search_queries", "sequence_number": 2, "queries": ["best running shoe brands"]],
            ["type": "response.reasoning.fetch_url_queries", "sequence_number": 3, "urls": ["https://example.com/source"]],
            ["type": "response.reasoning.stopped", "sequence_number": 4],
            ["type": "response.completed", "sequence_number": 5, "response": createResponse(["output": [messageItem]])]
        ]))
        #expect(finishPart(try await collect(model, hello))?.metadata["perplexity"]?["searchQueries"] == ["running shoes Denmark", "best running shoe brands"])
    }

    @Test(arguments: ["response.completed", "response.incomplete"])
    func preservesCitationAnnotationsFromOutputItems(_ terminalType: String) async throws {
        let message: JSONValue = ["id": "msg-123", "type": "message", "content": [["type": "output_text", "text": "A cited answer.", "annotations": [["type": "url_citation", "url": "https://example.com/first", "title": "First"], ["type": "file_citation", "file_id": "file-123"]]]]]
        let second: JSONValue = ["type": "message", "content": [["type": "output_text", "text": "Another citation.", "annotations": [["type": "url_citation", "url": "https://example.com/second", "title": "Second"]]]]]
        let (model, _) = try makeModel(stream([
            ["type": "response.output_item.done", "item": message, "output_index": 0],
            ["type": .string(terminalType), "response": createResponse(["status": terminalType == "response.incomplete" ? "incomplete" : "completed", "output": [message, second]])]
        ]))
        let sources = try await collect(model, hello).compactMap { part -> [String?]? in if case let .source(source) = part { [source.url, source.title] } else { nil } }
        #expect(sources == [["https://example.com/first", "First"], ["https://example.com/second", "Second"]])
    }

    @Test func preservesNativeToolTracesInRawChunksWithoutRejectingTheResponse() async throws {
        let events = [["type": "response.output_item.done", "output_index": 0, "item": financeOutput] as JSONValue] + createStreamChunks(["output": [financeOutput, searchResultsItem, messageItem]])
        let (model, _) = try makeModel(stream(events))
        let parts = try await collect(model, LanguageModelRequest(messages: [.user("Hello")], includeRawChunks: true))
        #expect(!parts.contains { if case .error = $0 { true } else { false } })
        #expect(parts.contains { if case let .raw(value) = $0 { value == events[0] } else { false } })
        #expect(finishPart(parts)?.reason == "stop")
    }

    @Test func streamsTypedAgentAPIEventsAsTextSourcesUsageAndMetadata() async throws {
        let (model, transport) = try makeModel(stream(createStreamChunks()))
        let parts = try await collect(model, hello)
        let summary = parts.compactMap { part -> String? in
            switch part {
            case let .streamStart(warnings): "stream-start:\(warnings.count)"
            case let .responseMetadata(metadata): "response-metadata:\(metadata.id ?? ""):\(metadata.modelID ?? ""):\(metadata.timestamp?.timeIntervalSince1970 ?? 0)"
            case let .source(source): "source:\(source.url ?? "")"
            case let .textStart(id, _): "text-start:\(id)"
            case let .textDeltaPart(id, delta, _): "text-delta:\(id):\(delta)"
            case let .textEnd(id, _): "text-end:\(id)"
            case let .finishMetadata(reason, usage, _): "finish:\(reason ?? ""):\(usage?.inputTokens ?? -1):\(usage?.outputTokens ?? -1)"
            default: nil
            }
        }
        #expect(summary == [
            "stream-start:0", "response-metadata:resp-123:openai/gpt-5.1:1784292159.0", "source:https://example.com/source",
            "text-start:msg-123", "text-delta:msg-123:Hello ", "text-delta:msg-123:from Perplexity.", "text-end:msg-123", "finish:stop:120:45"
        ])
        #expect(try await sentBody(transport) == ["preset": "low", "input": [["type": "message", "role": "user", "content": "Hello"]], "stream": true])
    }

    @Test func streamsRawAgentAPIEventsWhenRequested() async throws {
        let events = createStreamChunks()
        let (model, _) = try makeModel(stream(events))
        let raw = try await collect(model, LanguageModelRequest(messages: [.user("Hello")], includeRawChunks: true)).compactMap { part -> JSONValue? in if case let .raw(value) = part { value } else { nil } }
        #expect(raw == events)
    }

    @Test func emitsRawChunksBeforeSchemaValidation() async throws {
        let invalid: JSONValue = ["type": "response.output_text.delta", "delta": 1]
        let (model, _) = try makeModel(stream([invalid]))
        let parts = try await collect(model, LanguageModelRequest(messages: [.user("Hello")], includeRawChunks: true))
        #expect(parts.contains { if case let .raw(value) = $0 { value == invalid } else { false } })
        #expect(parts.contains { if case .error = $0 { true } else { false } })
        #expect(finishPart(parts)?.reason == "error")
    }

    @Test func streamsFunctionCallsFromOutputItems() async throws {
        let functionCall: JSONValue = ["id": "fc-123", "type": "function_call", "status": "completed", "call_id": "call-123", "name": "weather", "arguments": #"{"city":"San Francisco"}"#]
        let (model, _) = try makeModel(stream([
            ["type": "response.created", "sequence_number": 0, "response": createResponse(["output": JSONValue.array([JSONValue]()), "usage": nil])],
            ["type": "response.output_item.done", "sequence_number": 1, "output_index": 0, "item": functionCall],
            ["type": "response.completed", "sequence_number": 2, "response": createResponse(["status": "requires_action", "output": [functionCall]])]
        ]))
        let parts = try await collect(model, hello)
        let summary = parts.compactMap { part -> String? in
            switch part {
            case let .toolInputStart(id, name, _, _, _, _): "start:\(id):\(name)"
            case let .toolInputDelta(id, delta, _): "delta:\(id):\(delta)"
            case let .toolInputEnd(id, _): "end:\(id)"
            case let .toolCall(call): "call:\(call.id):\(call.name)"
            default: nil
            }
        }
        #expect(summary == ["start:call-123:weather", #"delta:call-123:{"city":"San Francisco"}"#, "end:call-123", "call:call-123:weather"])
        #expect(finishPart(parts)?.reason == "tool-calls")
    }

    @Test func streamsSourcesFromFetchURLReasoningEvents() async throws {
        let (model, _) = try makeModel(stream([["type": "response.reasoning.fetch_url_results", "sequence_number": 0, "contents": [["title": "Fetched page", "url": "https://example.com/fetched", "snippet": "Fetched content."]]]]))
        let sources = try await collect(model, hello).compactMap { part -> AISource? in if case let .source(source) = part { source } else { nil } }
        #expect(sources.map(\.url) == ["https://example.com/fetched"])
        #expect(sources.first?.title == "Fetched page")
    }

    @Test func emitsEmptyTextDeltasLikeUpstream() async throws {
        let (model, _) = try makeModel(stream([
            ["type": "response.output_text.delta", "item_id": "msg-1", "output_index": 0, "content_index": 0, "delta": ""],
            ["type": "response.completed"]
        ]))
        let summary = try await collect(model, hello).compactMap { part -> String? in
            switch part {
            case let .textStart(id, _): "start:\(id)"
            case let .textDeltaPart(id, delta, _): "delta:\(id):\(delta)"
            case let .textEnd(id, _): "end:\(id)"
            default: nil
            }
        }
        #expect(summary == ["start:msg-1", "delta:msg-1:", "end:msg-1"])
    }

    @Test func streamsResponseFailedAsAnError() async throws {
        let (model, _) = try makeModel(stream([["type": "response.failed", "error": ["message": "Agent run failed"]]]))
        let parts = try await collect(model, hello)
        #expect(parts.contains { if case let .error(message, _) = $0 { message == "Agent run failed" } else { false } })
        #expect(finishPart(parts)?.reason == "error")
    }
}

private func textParts(_ parts: [LanguageStreamPart]) -> [String] {
    parts.compactMap { part -> String? in
        switch part {
        case let .textStart(id, _): "start:\(id)"
        case let .textDeltaPart(id, delta, _): "delta:\(id):\(delta)"
        case let .textEnd(id, _): "end:\(id)"
        default: nil
        }
    }
}

@Suite("Perplexity upstream doStream text and sources")
struct PerplexityUpstreamStreamTextTests {
    @Test func acceptsNullFieldsInStreamEvents() async throws {
        let (model, _) = try makeModel(stream([
            ["type": "response.reasoning.started", "sequence_number": 0, "thought": .null],
            ["type": "response.reasoning.fetch_url_results", "call_id": "call-1", "sequence_number": 1, "thought": "Fetched content from 0 URLs", "contents": .null],
            ["type": "response.reasoning.search_results", "sequence_number": 2, "results": .null],
            ["type": "response.reasoning.stopped", "sequence_number": 3],
            ["type": "response.output_text.delta", "item_id": "msg-1", "output_index": 0, "content_index": .null, "delta": "Hello"],
            ["type": "response.output_text.done", "item_id": "msg-1", "output_index": 0, "content_index": .null, "text": .null]
        ]))
        let parts = try await collect(model, hello)
        #expect(!parts.contains { if case .error = $0 { true } else { false } })
        let reasoning = parts.compactMap { part -> String? in
            switch part {
            case let .reasoningStart(id, _): "start:\(id)"
            case let .reasoningDeltaPart(id, delta, _): "delta:\(id):\(delta)"
            case let .reasoningEnd(id, _): "end:\(id)"
            default: nil
            }
        }
        #expect(reasoning == ["start:reasoning-0", "delta:reasoning-0:Fetched content from 0 URLs", "end:reasoning-0"])
        #expect(textParts(parts) == ["start:msg-1", "delta:msg-1:Hello", "end:msg-1"])
    }

    @Test func emitsAFetchedURLOnlyOnceWithItsLaterSearchResultID() async throws {
        let (model, _) = try makeModel(stream([
            ["type": "response.reasoning.fetch_url_results", "sequence_number": 0, "contents": [["title": "Fetched page", "url": "https://example.com/source", "snippet": "Fetched content."]]],
            ["type": "response.reasoning.search_results", "sequence_number": 1, "results": [["id": 7, "title": "Search result", "url": "https://example.com/source", "snippet": "Search result content."]]]
        ]))
        let sources = try await collect(model, hello).compactMap { part -> AISource? in if case let .source(source) = part { source } else { nil } }
        #expect(sources.count == 1)
        #expect(sources.first?.id == "7" && sources.first?.providerMetadata["perplexity"]?["resultId"] == 7)
    }

    @Test func deduplicatesAURLAcrossAnnotationsSearchResultsAndFetchedContents() async throws {
        let searchResult: JSONValue = ["id": 7, "title": "Search result", "url": "https://example.com/source", "snippet": "Search result content."]
        let message: JSONValue = ["type": "message", "content": [["type": "output_text", "text": "Answer", "annotations": [["type": "url_citation", "url": "https://example.com/source"]]]]]
        let (model, _) = try makeModel(stream([
            ["type": "response.output_item.done", "item": message],
            ["type": "response.reasoning.search_results", "results": [searchResult]],
            ["type": "response.reasoning.fetch_url_results", "contents": [searchResult]],
            ["type": "response.completed", "response": createResponse(["output": [["type": "search_results", "results": [searchResult]], message]])]
        ]))
        let sources = try await collect(model, hello).compactMap { part -> AISource? in if case let .source(source) = part { source } else { nil } }
        #expect(sources.map(\.id) == ["7"])
    }

    @Test func streamsAgentAPIReasoningThoughts() async throws {
        let (model, _) = try makeModel(stream([
            ["type": "response.reasoning.started", "sequence_number": 0, "thought": "Planning. "],
            ["type": "response.reasoning.search_queries", "sequence_number": 1, "queries": ["latest AI news"], "thought": "Searching. "],
            ["type": "response.reasoning.search_results", "sequence_number": 2, "results": JSONValue.array([JSONValue]()), "thought": "Reviewing results. "],
            ["type": "response.reasoning.fetch_url_queries", "sequence_number": 3, "urls": ["https://example.com/source"], "thought": "Fetching details. "],
            ["type": "response.reasoning.fetch_url_results", "sequence_number": 4, "contents": JSONValue.array([JSONValue]()), "thought": "Checking details. "],
            ["type": "response.reasoning.stopped", "sequence_number": 5, "thought": "Done."]
        ]))
        let parts = try await collect(model, hello)
        let summary = parts.compactMap { part -> String? in
            switch part {
            case .streamStart: "stream-start"
            case let .reasoningStart(id, _): "start:\(id)"
            case let .reasoningDeltaPart(_, delta, _): delta
            case let .reasoningEnd(id, _): "end:\(id)"
            case .finishMetadata: "finish"
            default: nil
            }
        }
        #expect(summary == ["stream-start", "start:reasoning-0", "Planning. ", "Searching. ", "Reviewing results. ", "Fetching details. ", "Checking details. ", "Done.", "end:reasoning-0", "finish"])
    }

    @Test func handlesIncompleteTerminalEventsAndPreservesUsage() async throws {
        let (model, _) = try makeModel(stream([["type": "response.incomplete", "sequence_number": 0, "response": createResponse(["status": "incomplete", "incomplete_details": ["reason": "max_output_tokens"]])]]))
        let parts = try await collect(model, hello)
        #expect(textParts(parts) == ["start:msg-123", "delta:msg-123:Hello from Perplexity.", "end:msg-123"])
        let finish = finishPart(parts)
        #expect(finish?.reason == "length" && finish?.usage?.inputTokens == 120 && finish?.usage?.outputTokens == 45)
    }

    @Test(arguments: ["response.output_text.done", "response.output_item.done", "response.completed"])
    func recoversTextWithoutDeltas(_ type: String) async throws {
        let response = createResponse(["output": [messageItem]])
        let (model, _) = try makeModel(stream([["type": .string(type), "item_id": "msg-123", "output_index": 0, "content_index": 0, "text": "Hello from Perplexity.", "item": messageItem, "response": response]]))
        #expect(textParts(try await collect(model, hello)) == ["start:msg-123", "delta:msg-123:Hello from Perplexity.", "end:msg-123"])
    }

    @Test func appendsOnlyMissingTextAndDoesNotRepeatCompletedContentParts() async throws {
        let message: JSONValue = ["type": "message", "id": "msg-123", "content": [["type": "output_text", "text": "Hello world."], ["type": "output_text", "text": "Second part."]]]
        let (model, _) = try makeModel(stream([
            ["type": "response.output_text.delta", "item_id": "msg-123", "content_index": 0, "delta": "Hello "],
            ["type": "response.output_text.done", "item_id": "msg-123", "content_index": 0, "text": "Hello world."],
            ["type": "response.output_text.delta", "item_id": "msg-123", "content_index": 1, "delta": "Second "],
            ["type": "response.output_item.done", "item": message, "output_index": 0],
            ["type": "response.completed", "response": createResponse(["output": [message]])]
        ]))
        #expect(textParts(try await collect(model, hello)) == [
            "start:msg-123", "delta:msg-123:Hello ", "delta:msg-123:world.", "end:msg-123",
            "start:msg-123:1", "delta:msg-123:1:Second ", "delta:msg-123:1:part.", "end:msg-123:1"
        ])
    }

    @Test func closesUnfinishedTextPartsInStartOrder() async throws {
        let ids = (0..<8).map { "msg-\($0)" }
        let (model, _) = try makeModel(stream(ids.map { ["type": "response.output_text.delta", "item_id": .string($0), "delta": "x"] }))
        let ends = textParts(try await collect(model, hello)).filter { $0.hasPrefix("end:") }
        #expect(ends == ids.map { "end:\($0)" })
    }
}

@Suite("Perplexity upstream decisions")
struct PerplexityUpstreamDecisionTests {
    private static let fixture: JSONValue = [
        "model": "pplx-decider-latest",
        "answers": [
            "defect": ["type": "noul", "noul": 0.9424522889347015],
            "sentiment": ["type": "choice", "choice": "mixed", "confidence": 0.9255246944002182, "probabilities": ["positive": 0.020649883775315993, "mixed": 0.9503497962668123, "negative": 0.02900031995787183]],
            "severity": ["type": "score", "score": 1.7838686319784252, "confidence": 0.7838686319784252, "legend": ["0": "Cosmetic", "1": "Inconvenient", "2": "Product unusable"], "probabilities": ["0": 0.008423954913615923, "1": 0.199283458194343, "2": 0.7922925868920411]]
        ],
        "usage": ["input_tokens": 367, "output_tokens": 3]
    ]
    private static let questions: [String: AIDecisionQuestion] = [
        "defect": .boolean(instructions: "Does the review report a defect?"),
        "sentiment": .choice(instructions: "What is the sentiment?", criteria: ["positive": "Satisfied", "mixed": "Praise and complaints", "negative": "Dissatisfied"]),
        "severity": .score(instructions: "How severe is the problem?", criteria: ["Cosmetic", "Inconvenient", "Product unusable"])
    ]

    private func setup(_ body: JSONValue = fixture, status: Int = 200, headers: [String: String] = [:]) throws -> (any AIDecisionModelV4, RecordingTransport) {
        let response = AIHTTPResponse(statusCode: status, headers: ["content-type": "application/json", "x-request-id": "request-123", "retry-after": "2"].merging(headers) { _, new in new }, body: (try? encodeJSONBody(body)) ?? Data())
        let transport = RecordingTransport(response: response)
        let provider = try PerplexityProvider(configuration: .init(apiKey: "test-key", baseURL: "https://example.com/", headers: ["x-provider": "custom"], transport: transport))
        return (try provider.decisionModel("pplx-decider-latest"), transport)
    }

    private var options: AIDecisionModelV4CallOptions {
        .init(state: [.json(["review": "Great sound, but the battery broke."])], questions: Self.questions)
    }

    @Test func sendsOneNativeRequestWithAllQuestionTypes() async throws {
        let (model, transport) = try setup()
        _ = try await model.doDecide(options)
        let request = try #require(await transport.requests().first)
        #expect(request.url.absoluteString == "https://example.com/v1/decisions")
        #expect(request.headers["authorization"] == "Bearer test-key")
        #expect(request.headers["x-provider"] == "custom")
        #expect(request.headers["X-Pplx-Integration"] ?? request.headers["x-pplx-integration"] == "vercel-ai-sdk")
        #expect(try decodeJSONBody(try #require(request.body)) == [
            "model": "pplx-decider-latest",
            "state": ["review": "Great sound, but the battery broke."],
            "questions": [
                "defect": ["type": "noul", "instructions": "Does the review report a defect?"],
                "sentiment": ["type": "choice", "instructions": "What is the sentiment?", "criteria": ["positive": "Satisfied", "mixed": "Praise and complaints", "negative": "Dissatisfied"]],
                "severity": ["type": "score", "instructions": "How severe is the problem?", "criteria": ["Cosmetic", "Inconvenient", "Product unusable"]]
            ]
        ])
    }

    @Test func preservesNativePrecisionConfidenceLegendsUsageAndResponseMetadata() async throws {
        let (model, _) = try setup()
        let result = try await model.doDecide(options)
        #expect(result.answers["defect"] == .boolean(probability: 0.9424522889347015))
        #expect(result.answers["sentiment"] == .choice(choice: "mixed", probabilities: ["positive": 0.020649883775315993, "mixed": 0.9503497962668123, "negative": 0.02900031995787183]))
        #expect(result.usage == .init(inputTokens: 367, outputTokens: 3))
        #expect(result.providerMetadata == ["perplexity": [
            "confidence": ["sentiment": 0.9255246944002182, "severity": 0.7838686319784252],
            "legend": ["severity": ["0": "Cosmetic", "1": "Inconvenient", "2": "Product unusable"]]
        ]])
        #expect(result.response?.id == "request-123" && result.response?.modelID == "pplx-decider-latest")
    }

    @Test func warnsAboutUnsupportedOptionsWithoutAddingFieldsToTheRequest() async throws {
        let (model, transport) = try setup()
        var call = options
        call.providerOptions = ["perplexity": ["temperature": 0, "searchDomainFilter": ["example.com"]]]
        let result = try await model.doDecide(call)
        #expect(Set(result.warnings.compactMap(\.feature)) == ["providerOptions.perplexity.temperature", "providerOptions.perplexity.searchDomainFilter"])
        #expect(result.warnings.allSatisfy { $0.type == "unsupported" })
        #expect(Set(try await sentBody(transport).objectValue?.keys.map { $0 } ?? []) == ["model", "state", "questions"])
    }

    @Test(arguments: [[:], ["extra": ["type": "noul", "noul": 0.5]]] as [[String: JSONValue]])
    func rejectsMissingAndUnexpectedAnswers(_ change: [String: JSONValue]) async throws {
        var answers = change.isEmpty ? [:] : (Self.fixture["answers"]?.objectValue ?? [:])
        answers.merge(change) { _, new in new }
        let (model, _) = try setup(["answers": .object(answers)])
        await #expect(throws: AIError.self) { _ = try await model.doDecide(options) }
    }

    @Test(arguments: [["defect": ["type": "noul", "noul": 1.01]], ["sentiment": ["type": "choice", "choice": "mixed"]], ["defect": ["type": "refusal"]]] as [[String: JSONValue]])
    func rejectsMalformedNativeResponses(_ change: [String: JSONValue]) async throws {
        var answers = Self.fixture["answers"]?.objectValue ?? [:]
        answers.merge(change) { _, new in new }
        let (model, _) = try setup(["answers": .object(answers)])
        await #expect(throws: AIError.self) { _ = try await model.doDecide(options) }
    }

    @Test func toleratesNullAncillaryResponseFields() async throws {
        var body = Self.fixture.objectValue ?? [:]
        body["model"] = .null
        body["usage"] = .null
        let (model, _) = try setup(.object(body))
        let result = try await model.doDecide(options)
        #expect(result.usage == nil)
        #expect(result.response?.modelID == "pplx-decider-latest")
    }

    @Test(arguments: [JSONValue.null, "400", 401])
    func preservesErrorMessagesWithCode(_ code: JSONValue) async throws {
        let (model, _) = try setup(["error": ["code": code, "message": "Invalid model", "type": "invalid_request_error", "param": .null]], status: 400)
        do {
            _ = try await model.doDecide(options)
            Issue.record("expected failure")
        } catch AIError.apiCall(let error) {
            #expect(error.message == "Invalid model")
            #expect(error.statusCode == 400)
            #expect(!error.isRetryable)
        }
    }

    @Test(arguments: [404, 405, 413, 429, 500, 502, 503, 504])
    func preservesHTTPStatusAndRetryHeadersEvenWithNonJSONErrors(_ status: Int) async throws {
        let response = AIHTTPResponse(statusCode: status, headers: ["retry-after": "2"], body: Data((status == 504 ? "<html>Gateway Timeout</html>" : "").utf8))
        let transport = RecordingTransport(response: response)
        let model = try PerplexityProvider(configuration: .init(apiKey: "test-key", transport: transport)).decisionModel("pplx-decider-latest")
        do {
            _ = try await model.doDecide(options)
            Issue.record("expected failure")
        } catch AIError.apiCall(let error) {
            #expect(error.statusCode == status)
            #expect(error.isRetryable == (status == 429 || status >= 500))
            #expect(error.responseHeaders["retry-after"] == "2")
        }
    }
}

@Suite("Perplexity upstream usage conversion")
struct PerplexityUpstreamUsageTests {
    @Test func treatsReasoningTokensAsSeparateFromCompletionTokens() async throws {
        let usage: JSONValue = ["input_tokens": 33, "output_tokens": 205342, "total_tokens": 205375, "output_tokens_details": ["reasoning_tokens": 193947]]
        let (model, _) = try makeModel(json(createResponse(["usage": usage])))
        let result = try await model.generate(hello)
        #expect(result.usage?.inputTokens == 33 && result.usage?.inputTokensNoCache == 33 && result.usage?.inputTokensCacheRead == 0 && result.usage?.inputTokensCacheWrite == 0)
        #expect(result.usage?.outputTokens == 205342 && result.usage?.outputTextTokens == 11395 && result.usage?.outputReasoningTokens == 193947)
        #expect(result.usage?.rawValue == usage)
    }
}
