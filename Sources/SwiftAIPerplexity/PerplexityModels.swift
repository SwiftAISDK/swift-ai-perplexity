import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class PerplexityLanguageModel: LanguageModel, @unchecked Sendable {
    public let providerID = "perplexity"
    public let modelID: String
    private let config: PerplexityModelConfiguration

    public init(modelID: String, config: PerplexityModelConfiguration) {
        self.modelID = modelID
        self.config = config
    }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        let prepared = try perplexityAgentPreparedCall(for: request, modelID: modelID, stream: false)
        let body = config.transformRequestBody?(prepared.body) ?? prepared.body
        let response = try await config.sendJSONResponse(
            path: "/v1/agent",
            modelID: modelID,
            body: .object(body),
            headers: request.headers,
            abortSignal: request.abortSignal
        )
        let raw = response.json
        try validatePerplexityAgentResponse(raw, providerID: providerID)
        if raw["status"]?.stringValue == "failed" || raw["error"] != nil && raw["error"] != .null {
            throw AIError.apiCall(AIAPICallError(
                provider: providerID,
                url: response.response.url?.absoluteString,
                requestBody: .object(body),
                statusCode: 400,
                responseHeaders: response.response.headers,
                responseBody: response.response.bodyText,
                isRetryable: false
            ))
        }

        var content: [AIResultContentPart] = []
        var sourceIndexes: [String: Int] = [:]
        var hasFunctionCall = false
        for item in raw["output"]?.arrayValue ?? [] {
            perplexityAgentAppendOutputItem(
                item,
                content: &content,
                sourceIndexes: &sourceIndexes,
                hasFunctionCall: &hasFunctionCall
            )
        }
        let text = content.compactMap { part -> String? in
            if case let .text(value, _) = part { return value }
            return nil
        }.joined()
        let finishReason = perplexityAgentFinishReason(
            status: raw["status"]?.stringValue,
            incompleteReason: raw["incomplete_details"]?["reason"]?.stringValue,
            hasFunctionCall: hasFunctionCall
        )
        return TextGenerationResult(
            text: text,
            content: content,
            finishReason: finishReason,
            usage: perplexityAgentUsage(from: raw["usage"]),
            providerMetadata: perplexityAgentProviderMetadata(from: raw["usage"]),
            rawValue: raw,
            warnings: prepared.warnings,
            requestMetadata: aiRequestMetadata(body: .object(body), headers: request.headers),
            responseMetadata: perplexityAgentResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }

    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let prepared = try perplexityAgentPreparedCall(for: request, modelID: modelID, stream: true)
                    let body = JSONValue.object(config.transformRequestBody?(prepared.body) ?? prepared.body)
                    let httpRequest = try config.request(
                        path: "/v1/agent",
                        modelID: modelID,
                        body: body,
                        headers: request.headers,
                        abortSignal: request.abortSignal
                    )
                    let response = try await config.streamRequest(httpRequest)
                    guard (200..<300).contains(response.statusCode) else {
                        throw apiCallError(provider: providerID, response: try await bufferedHTTPResponse(from: response, request: httpRequest))
                    }
                    let responseHead = httpResponseHead(from: response, request: httpRequest)
                    continuation.yield(.streamStart(warnings: prepared.warnings))

                    var finishReason: String? = "interrupted"
                    var usage: TokenUsage?
                    var providerMetadata = perplexityAgentProviderMetadata(from: nil)
                    var hasResponseMetadata = false
                    var activeReasoningID: String?
                    var textByID: [String: String] = [:]
                    var endedTextIDs: Set<String> = []
                    var emittedSourceURLs: Set<String> = []
                    var pendingSourcesByURL: [String: AISource] = [:]
                    var seenFunctionCalls: Set<String> = []
                    var hasFunctionCall = false

                    func textID(_ raw: JSONValue, item: JSONValue? = nil) -> String {
                        let base = raw["item_id"]?.stringValue
                            ?? item?["id"]?.stringValue
                            ?? raw["output_index"]?.intValue.map { String($0) }
                            ?? "text"
                        let contentIndex = raw["content_index"]?.intValue ?? 0
                        return contentIndex == 0 ? base : "\(base):\(contentIndex)"
                    }

                    func emitTextDelta(id: String, delta: String) {
                        guard !endedTextIDs.contains(id), !delta.isEmpty else { return }
                        if textByID[id] == nil {
                            textByID[id] = ""
                            continuation.yield(.textStart(id: id))
                        }
                        textByID[id, default: ""] += delta
                        continuation.yield(.textDeltaPart(id: id, delta: delta))
                    }

                    func finishText(id: String, finalText: String?) {
                        guard !endedTextIDs.contains(id) else { return }
                        let emitted = textByID[id] ?? ""
                        if let finalText, finalText.hasPrefix(emitted), finalText.count > emitted.count {
                            emitTextDelta(id: id, delta: String(finalText.dropFirst(emitted.count)))
                        }
                        guard textByID[id] != nil else { return }
                        endedTextIDs.insert(id)
                        continuation.yield(.textEnd(id: id))
                    }

                    func emitSource(_ source: AISource) {
                        guard let url = source.url, !emittedSourceURLs.contains(url) else { return }
                        if perplexityAgentSourceHasResultID(source) {
                            pendingSourcesByURL.removeValue(forKey: url)
                            emittedSourceURLs.insert(url)
                            continuation.yield(.source(source))
                        } else if pendingSourcesByURL[url] == nil {
                            pendingSourcesByURL[url] = source
                        }
                    }

                    func emitFunctionCall(_ item: JSONValue) {
                        guard item["type"]?.stringValue == "function_call",
                              let callID = item["call_id"]?.stringValue,
                              let name = item["name"]?.stringValue,
                              let arguments = item["arguments"]?.stringValue,
                              !seenFunctionCalls.contains(callID) else { return }
                        seenFunctionCalls.insert(callID)
                        hasFunctionCall = true
                        let call = perplexityAgentToolCall(from: item, callID: callID, name: name, arguments: arguments)
                        continuation.yield(.toolInputStart(id: callID, name: name))
                        continuation.yield(.toolInputDelta(id: callID, delta: arguments))
                        continuation.yield(.toolInputEnd(id: callID))
                        continuation.yield(.toolCall(call))
                    }

                    func emitOutputItem(_ item: JSONValue, outputIndex: Int?) {
                        switch item["type"]?.stringValue {
                        case "message":
                            for (contentIndex, part) in (item["content"]?.arrayValue ?? []).enumerated() {
                                if part["type"]?.stringValue == "output_text" {
                                    let synthetic: JSONValue = .object([
                                        "item_id": item["id"] ?? .null,
                                        "output_index": outputIndex.map { .number(Double($0)) } ?? .null,
                                        "content_index": .number(Double(contentIndex))
                                    ])
                                    finishText(id: textID(synthetic, item: item), finalText: part["text"]?.stringValue)
                                }
                                for annotation in part["annotations"]?.arrayValue ?? [] {
                                    if let source = perplexityAgentAnnotationSource(annotation) { emitSource(source) }
                                }
                            }
                        case "search_results":
                            for result in item["results"]?.arrayValue ?? [] {
                                if let source = perplexityAgentSearchSource(result) { emitSource(source) }
                            }
                        case "fetch_url_results":
                            for result in item["contents"]?.arrayValue ?? [] {
                                if let source = perplexityAgentFetchedSource(result) { emitSource(source) }
                            }
                        case "function_call":
                            emitFunctionCall(item)
                        default:
                            break
                        }
                    }

                    for try await event in serverSentEvents(from: response.body) {
                        if event.data == "[DONE]" { break }
                        let raw: JSONValue
                        do {
                            raw = try decodeJSONBody(Data(event.data.utf8))
                        } catch {
                            finishReason = "error"
                            continuation.yield(.error(message: String(describing: error)))
                            continue
                        }
                        do {
                            try validatePerplexityAgentChunk(raw, providerID: providerID)
                        } catch {
                            finishReason = "error"
                            continuation.yield(.error(message: String(describing: error), rawValue: raw))
                            continue
                        }
                        if request.includeRawChunks { continuation.yield(.raw(raw)) }
                        switch raw["type"]?.stringValue {
                        case "response.created", "response.in_progress":
                            if !hasResponseMetadata, let responseValue = raw["response"], responseValue != .null {
                                hasResponseMetadata = true
                                continuation.yield(.responseMetadata(perplexityAgentResponseMetadata(from: responseValue, response: responseHead, modelID: modelID)))
                            }
                        case "response.output_text.delta":
                            if let delta = raw["delta"]?.stringValue { emitTextDelta(id: textID(raw), delta: delta) }
                        case "response.output_text.done":
                            finishText(id: textID(raw), finalText: raw["text"]?.stringValue)
                        case "response.reasoning.started":
                            if let activeReasoningID { continuation.yield(.reasoningEnd(id: activeReasoningID)) }
                            let id = "reasoning-\(raw["sequence_number"]?.intValue.map { String($0) } ?? generateId())"
                            activeReasoningID = id
                            continuation.yield(.reasoningStart(id: id))
                            if let thought = raw["thought"]?.stringValue { continuation.yield(.reasoningDeltaPart(id: id, delta: thought)) }
                        case "response.reasoning.search_queries", "response.reasoning.fetch_url_queries":
                            if let id = activeReasoningID, let thought = raw["thought"]?.stringValue {
                                continuation.yield(.reasoningDeltaPart(id: id, delta: thought))
                            }
                        case "response.reasoning.search_results":
                            if let id = activeReasoningID, let thought = raw["thought"]?.stringValue {
                                continuation.yield(.reasoningDeltaPart(id: id, delta: thought))
                            }
                            for result in raw["results"]?.arrayValue ?? [] {
                                if let source = perplexityAgentSearchSource(result) { emitSource(source) }
                            }
                        case "response.reasoning.fetch_url_results":
                            if let id = activeReasoningID, let thought = raw["thought"]?.stringValue {
                                continuation.yield(.reasoningDeltaPart(id: id, delta: thought))
                            }
                            for result in raw["contents"]?.arrayValue ?? [] {
                                if let source = perplexityAgentFetchedSource(result) { emitSource(source) }
                            }
                        case "response.reasoning.stopped":
                            if let id = activeReasoningID {
                                if let thought = raw["thought"]?.stringValue { continuation.yield(.reasoningDeltaPart(id: id, delta: thought)) }
                                continuation.yield(.reasoningEnd(id: id))
                                activeReasoningID = nil
                            }
                        case "response.output_item.done":
                            if let item = raw["item"], item != .null { emitOutputItem(item, outputIndex: raw["output_index"]?.intValue) }
                        case "response.completed", "response.incomplete":
                            if let responseValue = raw["response"], responseValue != .null {
                                if !hasResponseMetadata {
                                    hasResponseMetadata = true
                                    continuation.yield(.responseMetadata(perplexityAgentResponseMetadata(from: responseValue, response: responseHead, modelID: modelID)))
                                }
                                for (index, item) in (responseValue["output"]?.arrayValue ?? []).enumerated() {
                                    emitOutputItem(item, outputIndex: index)
                                }
                                usage = perplexityAgentUsage(from: responseValue["usage"])
                                providerMetadata = perplexityAgentProviderMetadata(from: responseValue["usage"])
                                finishReason = perplexityAgentFinishReason(
                                    status: responseValue["status"]?.stringValue,
                                    incompleteReason: responseValue["incomplete_details"]?["reason"]?.stringValue,
                                    hasFunctionCall: hasFunctionCall
                                )
                            } else if finishReason == "interrupted" {
                                finishReason = "other"
                            }
                        case "response.failed":
                            finishReason = "error"
                            continuation.yield(.error(message: raw["error"]?["message"]?.stringValue ?? "Perplexity response failed", rawValue: raw))
                        default:
                            break
                        }
                    }
                    for source in pendingSourcesByURL.values { continuation.yield(.source(source)) }
                    if let activeReasoningID { continuation.yield(.reasoningEnd(id: activeReasoningID)) }
                    for id in textByID.keys where !endedTextIDs.contains(id) { continuation.yield(.textEnd(id: id)) }
                    continuation.yield(.finishMetadata(reason: finishReason, usage: usage, providerMetadata: providerMetadata))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}

/// Compatibility surface for the pre-5.0 Sonar Chat Completions API.
public final class PerplexitySonarLanguageModel: LanguageModel, @unchecked Sendable {
    public let providerID = "perplexity"
    public let modelID: String
    private let config: PerplexityModelConfiguration

    public init(modelID: String, config: PerplexityModelConfiguration) {
        self.modelID = modelID
        self.config = config
    }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        let prepared = try perplexityPreparedCall(for: request, modelID: modelID, stream: false)
        let body = config.transformRequestBody?(prepared.body) ?? prepared.body
        let response = try await config.sendJSONResponse(
            path: "/chat/completions",
            modelID: modelID,
            body: .object(body),
            headers: request.headers,
            abortSignal: request.abortSignal
        )
        let raw = response.json
        try validatePerplexityGenerateResponse(raw, providerID: providerID)
        let choice = raw["choices"]?[0]
        guard let text = choice?["message"]?["content"]?.stringValue else {
            throw AIError.invalidResponse(provider: providerID, message: "No text content found in Perplexity response.")
        }
        return TextGenerationResult(
            text: text,
            finishReason: perplexityFinishReason(choice?["finish_reason"]?.stringValue),
            usage: perplexityUsage(from: raw),
            sources: perplexitySources(from: raw["citations"]),
            providerMetadata: perplexityProviderMetadata(from: raw),
            rawValue: raw,
            warnings: prepared.warnings,
            requestMetadata: aiRequestMetadata(body: .object(body), headers: request.headers),
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }

    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let prepared = try perplexityPreparedCall(for: request, modelID: modelID, stream: true)
                    let body = JSONValue.object(config.transformRequestBody?(prepared.body) ?? prepared.body)
                    let httpRequest = try config.request(path: "/chat/completions", modelID: modelID, body: body, headers: request.headers, abortSignal: request.abortSignal)
                    let response = try await config.streamRequest(httpRequest)
                    guard (200..<300).contains(response.statusCode) else {
                        throw apiCallError(provider: providerID, response: try await bufferedHTTPResponse(from: response, request: httpRequest))
                    }
                    let responseHead = httpResponseHead(from: response, request: httpRequest)

                    continuation.yield(.streamStart(warnings: prepared.warnings))
                    var latestUsage: TokenUsage?
                    var finishReason: String? = "other"
                    var providerMetadata = perplexityEmptyProviderMetadata()
                    var didEmitResponseMetadata = false
                    var didEmitSources = false
                    var activeTextID: String?
                    for try await event in serverSentEvents(from: response.body) {
                        if event.data == "[DONE]" { break }
                        let raw: JSONValue
                        do {
                            raw = try decodeJSONBody(Data(event.data.utf8))
                        } catch {
                            finishReason = "error"
                            continuation.yield(.error(message: String(describing: error)))
                            continue
                        }
                        if request.includeRawChunks {
                            continuation.yield(.raw(raw))
                        }
                        if !didEmitResponseMetadata {
                            didEmitResponseMetadata = true
                            continuation.yield(.responseMetadata(aiResponseMetadata(from: raw, response: responseHead, modelID: modelID)))
                        }
                        latestUsage = perplexityUsage(from: raw) ?? latestUsage
                        perplexityMergeProviderMetadata(from: raw, into: &providerMetadata)
                        if !didEmitSources, let citations = raw["citations"], citations != .null {
                            didEmitSources = true
                            for source in perplexitySources(from: citations) {
                                continuation.yield(.source(source))
                            }
                        }
                        if let delta = raw["choices"]?[0]?["delta"]?["content"]?.stringValue, !delta.isEmpty {
                            let id = activeTextID ?? "0"
                            if activeTextID == nil {
                                activeTextID = id
                                continuation.yield(.textStart(id: id))
                            }
                            continuation.yield(.textDeltaPart(id: id, delta: delta))
                        }
                        if let reason = raw["choices"]?[0]?["finish_reason"]?.stringValue {
                            finishReason = perplexityFinishReason(reason)
                        }
                    }
                    if let textID = activeTextID {
                        continuation.yield(.textEnd(id: textID))
                    }
                    continuation.yield(.finishMetadata(reason: finishReason, usage: latestUsage, providerMetadata: providerMetadata))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}

private struct PerplexityAgentPreparedCall {
    var body: [String: JSONValue]
    var warnings: [AIWarning]
}

private let perplexityAgentPresetIDs: Set<String> = ["fast", "low", "medium", "high", "xhigh"]

private func perplexityAgentPreparedCall(for request: LanguageModelRequest, modelID: String, stream: Bool) throws -> PerplexityAgentPreparedCall {
    var warnings: [AIWarning] = []
    if request.topK != nil { warnings.append(AIWarning(type: "unsupported", feature: "topK")) }
    if request.frequencyPenalty != nil { warnings.append(AIWarning(type: "unsupported", feature: "frequencyPenalty")) }
    if request.presencePenalty != nil { warnings.append(AIWarning(type: "unsupported", feature: "presencePenalty")) }
    if !request.stopSequences.isEmpty { warnings.append(AIWarning(type: "unsupported", feature: "stopSequences")) }
    if request.seed != nil { warnings.append(AIWarning(type: "unsupported", feature: "seed")) }

    let preparedInput = try perplexityAgentInput(from: request.messages)
    warnings.append(contentsOf: preparedInput.warnings)
    var options = try perplexityAgentProviderOptions(from: request)
    let nativeTools = options.removeValue(forKey: "tools")?.arrayValue ?? []
    var functionTools: [JSONValue] = []
    for (name, schema) in request.tools {
        if schema["type"]?.stringValue == "provider" {
            warnings.append(AIWarning(type: "unsupported", feature: "provider-defined tool \(schema["name"]?.stringValue ?? name)"))
            continue
        }
        var parameters = schema.objectValue
        let description = parameters?.removeValue(forKey: "description")?.stringValue
        let strict = parameters?.removeValue(forKey: "strict")?.boolValue
        parameters?.removeValue(forKey: "providerOptions")
        var tool: [String: JSONValue] = [
            "type": .string("function"),
            "name": .string(name),
            "parameters": parameters.map(JSONValue.object) ?? schema
        ]
        if let description { tool["description"] = .string(description) }
        if let strict { tool["strict"] = .bool(strict) }
        functionTools.append(.object(tool))
    }
    if let choice = request.toolChoice,
       choice.stringValue != "auto",
       choice["type"]?.stringValue != "auto" {
        warnings.append(AIWarning(
            type: "unsupported",
            feature: "toolChoice",
            message: "The Perplexity Agent API currently selects tools automatically."
        ))
    }

    if options["reasoning"] == nil, isCustomReasoning(request.reasoning) {
        if request.reasoning == "none" {
            warnings.append(AIWarning(type: "unsupported", feature: "reasoning \"none\""))
        } else if let reasoning = request.reasoning {
            if let effort = mapReasoningToProviderEffort(
                reasoning: reasoning,
                effortMap: ["minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "xhigh"],
                warnings: &warnings
            ) {
                options["reasoning"] = .object(["effort": .string(effort)])
            }
        }
    }

    var body = options
    body[perplexityAgentPresetIDs.contains(modelID) ? "preset" : "model"] = .string(modelID)
    body["input"] = .array(preparedInput.input)
    if let maxOutputTokens = request.maxOutputTokens { body["max_output_tokens"] = .number(Double(maxOutputTokens)) }
    if let temperature = request.temperature { body["temperature"] = .number(temperature) }
    if let topP = request.topP { body["top_p"] = .number(topP) }
    if stream { body["stream"] = .bool(true) }
    let tools = nativeTools + functionTools
    if !tools.isEmpty { body["tools"] = .array(tools) }
    if let responseFormat = request.responseFormat {
        switch responseFormat {
        case let .json(schema?, name, description):
            var jsonSchema: [String: JSONValue] = [
                "name": .string(name ?? "response"),
                "schema": schema,
                "strict": .bool(true)
            ]
            if let description {
                jsonSchema["description"] = .string(description)
            }
            body["response_format"] = .object([
                "type": .string("json_schema"),
                "json_schema": .object(jsonSchema)
            ])
        case .json(nil, _, _):
            warnings.append(AIWarning(type: "unsupported", feature: "JSON response format without a schema"))
        case .text:
            break
        }
    }
    return PerplexityAgentPreparedCall(body: body, warnings: warnings)
}

private func perplexityAgentProviderOptions(from request: LanguageModelRequest) throws -> [String: JSONValue] {
    var options = request.extraBody
    if let nested = options.removeValue(forKey: "perplexity")?.objectValue { options.merge(nested) { _, value in value } }
    if let value = request.providerOptions["perplexity"] {
        guard value != .null else { return options }
        guard let nested = value.objectValue else {
            throw AIError.invalidArgument(argument: "providerOptions.perplexity", message: "Perplexity provider options must be an object.")
        }
        options.merge(nested) { _, value in value }
    }
    let aliases = [
        "maxSteps": "max_steps", "maxToolCalls": "max_tool_calls",
        "previousResponseId": "previous_response_id", "languagePreference": "language_preference"
    ]
    for (source, destination) in aliases where options[destination] == nil {
        options[destination] = options.removeValue(forKey: source)
    }
    if let value = options["instructions"], value.stringValue == nil {
        throw AIError.invalidArgument(argument: "providerOptions.perplexity.instructions", message: "Perplexity instructions must be a string.")
    }
    if let value = options["models"], value.arrayValue?.allSatisfy({ $0.stringValue != nil }) != true {
        throw AIError.invalidArgument(argument: "providerOptions.perplexity.models", message: "Perplexity models must be an array of strings.")
    }
    if let value = options["max_steps"], (perplexityExactInteger(value) ?? 0) <= 0 {
        throw AIError.invalidArgument(argument: "providerOptions.perplexity.max_steps", message: "Perplexity max_steps must be a positive integer.")
    }
    if let value = options["max_tool_calls"], (perplexityExactInteger(value) ?? -1) < 0 {
        throw AIError.invalidArgument(argument: "providerOptions.perplexity.max_tool_calls", message: "Perplexity max_tool_calls must be a non-negative integer.")
    }
    if let value = options["previous_response_id"], value.stringValue == nil {
        throw AIError.invalidArgument(argument: "providerOptions.perplexity.previous_response_id", message: "Perplexity previous_response_id must be a string.")
    }
    if let value = options["store"], value.boolValue == nil {
        throw AIError.invalidArgument(argument: "providerOptions.perplexity.store", message: "Perplexity store must be a boolean.")
    }
    if let value = options["language_preference"], value.stringValue == nil {
        throw AIError.invalidArgument(argument: "providerOptions.perplexity.language_preference", message: "Perplexity language_preference must be a string.")
    }
    if let reasoning = options["reasoning"] {
        guard let object = reasoning.objectValue else {
            throw AIError.invalidArgument(argument: "providerOptions.perplexity.reasoning", message: "Perplexity reasoning must be an object.")
        }
        if let effort = object["effort"],
           effort.stringValue == nil
            || !["minimal", "low", "medium", "high", "xhigh"].contains(effort.stringValue ?? "") {
            throw AIError.invalidArgument(argument: "providerOptions.perplexity.reasoning.effort", message: "Perplexity reasoning effort is unsupported.")
        }
    }
    if let tools = options["tools"] {
        try validatePerplexityNativeTools(tools)
    }
    if let skills = options["skills"] {
        let builtinNames: Set<String> = ["office", "office/docx", "office/pdf", "office/pptx", "office/xlsx"]
        guard let values = skills.arrayValue else {
            throw AIError.invalidArgument(argument: "providerOptions.perplexity.skills", message: "Perplexity skills must be an array.")
        }
        for value in values {
            guard let object = value.objectValue, let type = object["type"]?.stringValue else {
                throw AIError.invalidArgument(argument: "providerOptions.perplexity.skills", message: "Each Perplexity skill must be a typed object.")
            }
            switch type {
            case "builtin":
                guard let name = object["name"]?.stringValue, builtinNames.contains(name) else {
                    throw AIError.invalidArgument(argument: "providerOptions.perplexity.skills", message: "Perplexity builtin skill name is unsupported.")
                }
            case "inline":
                guard object["name"]?.stringValue != nil,
                      object["description"]?.stringValue != nil,
                      object["instructions"]?.stringValue != nil else {
                    throw AIError.invalidArgument(argument: "providerOptions.perplexity.skills", message: "Perplexity inline skills require name, description, and instructions.")
                }
            default:
                throw AIError.invalidArgument(argument: "providerOptions.perplexity.skills", message: "Perplexity skill type is unsupported.")
            }
        }
    }
    return options
}

private func validatePerplexityNativeTools(_ value: JSONValue) throws {
    guard let tools = value.arrayValue else {
        throw perplexityNativeToolsError("Perplexity tools must be an array of typed tool objects.")
    }
    for tool in tools {
        guard let object = tool.objectValue, let type = object["type"]?.stringValue else {
            throw perplexityNativeToolsError("Each Perplexity tool must be a typed object.")
        }
        switch type {
        case "web_search":
            if let filters = object["filters"] {
                guard let filters = filters.objectValue else {
                    throw perplexityNativeToolsError("Perplexity web_search filters must be an object.")
                }
                if let domains = filters["search_domain_filter"], !perplexityStringArray(domains) {
                    throw perplexityNativeToolsError("Perplexity search_domain_filter must be an array of strings.")
                }
                if let recency = filters["search_recency_filter"] {
                    guard let value = recency.stringValue,
                          ["hour", "day", "week", "month", "year"].contains(value) else {
                        throw perplexityNativeToolsError("Perplexity search_recency_filter is unsupported.")
                    }
                }
                for key in ["search_after_date_filter", "search_before_date_filter", "last_updated_after_filter", "last_updated_before_filter"] {
                    if let entry = filters[key], entry.stringValue == nil {
                        throw perplexityNativeToolsError("Perplexity \(key) must be a string.")
                    }
                }
            }
            if let maxResults = object["max_results"], (perplexityExactInteger(maxResults) ?? 0) <= 0 {
                throw perplexityNativeToolsError("Perplexity web_search max_results must be a positive integer.")
            }
            for key in ["max_tokens", "max_tokens_per_page"] where object[key] != nil && object[key]?.doubleValue == nil {
                throw perplexityNativeToolsError("Perplexity web_search \(key) must be a number.")
            }
            if let contextSize = object["search_context_size"] {
                guard let value = contextSize.stringValue, ["low", "medium", "high"].contains(value) else {
                    throw perplexityNativeToolsError("Perplexity web_search search_context_size is unsupported.")
                }
            }
            if let location = object["user_location"] {
                guard let location = location.objectValue else {
                    throw perplexityNativeToolsError("Perplexity web_search user_location must be an object.")
                }
                for key in ["latitude", "longitude"] where location[key] != nil && location[key]?.doubleValue == nil {
                    throw perplexityNativeToolsError("Perplexity user_location \(key) must be a number.")
                }
                for key in ["country", "city", "region"] where location[key] != nil && location[key]?.stringValue == nil {
                    throw perplexityNativeToolsError("Perplexity user_location \(key) must be a string.")
                }
            }
        case "fetch_url":
            if let maxURLs = object["max_urls"], maxURLs.doubleValue == nil {
                throw perplexityNativeToolsError("Perplexity fetch_url max_urls must be a number.")
            }
        case "people_search", "finance_search", "sandbox":
            break
        case "mcp":
            guard object["server_label"]?.stringValue != nil,
                  object["server_url"]?.stringValue != nil else {
                throw perplexityNativeToolsError("Perplexity mcp tools require server_label and server_url strings.")
            }
            try validatePerplexityToolCommonFields(object)
            if let authorization = object["authorization"], authorization.stringValue == nil {
                throw perplexityNativeToolsError("Perplexity mcp authorization must be a string.")
            }
            if let deferLoading = object["defer_loading"], deferLoading.boolValue == nil {
                throw perplexityNativeToolsError("Perplexity mcp defer_loading must be a boolean.")
            }
            if let headers = object["headers"] {
                guard let headers = headers.objectValue, headers.values.allSatisfy({ $0.stringValue != nil }) else {
                    throw perplexityNativeToolsError("Perplexity mcp headers must map strings to strings.")
                }
            }
        case "connector":
            guard object["id"]?.stringValue != nil,
                  object["server_label"]?.stringValue != nil else {
                throw perplexityNativeToolsError("Perplexity connector tools require id and server_label strings.")
            }
            try validatePerplexityToolCommonFields(object)
            if let description = object["server_description"], description.stringValue == nil {
                throw perplexityNativeToolsError("Perplexity connector server_description must be a string.")
            }
        default:
            throw perplexityNativeToolsError("Perplexity native tool type \"\(type)\" is unsupported.")
        }
    }
}

private func validatePerplexityToolCommonFields(_ object: [String: JSONValue]) throws {
    if let allowedTools = object["allowed_tools"], !perplexityStringArray(allowedTools) {
        throw perplexityNativeToolsError("Perplexity allowed_tools must be an array of strings.")
    }
}

private func perplexityExactInteger(_ value: JSONValue) -> Int? {
    guard let number = value.doubleValue,
          number.isFinite,
          number.rounded(.towardZero) == number else { return nil }
    return Int(exactly: number)
}

private func perplexityStringArray(_ value: JSONValue) -> Bool {
    value.arrayValue?.allSatisfy { $0.stringValue != nil } == true
}

private func perplexityNativeToolsError(_ message: String) -> AIError {
    .invalidArgument(argument: "providerOptions.perplexity.tools", message: message)
}

private func perplexityAgentInput(from messages: [AIMessage]) throws -> (input: [JSONValue], warnings: [AIWarning]) {
    var input: [JSONValue] = []
    var warnings: [AIWarning] = []
    for message in messages {
        switch message.role {
        case .system:
            input.append(.object([
                "type": .string("message"),
                "role": .string("system"),
                "content": .string(message.content.compactMap(\.text).joined())
            ]))
        case .user:
            var parts: [JSONValue] = []
            for part in message.content {
                switch part {
                case let .text(text, _):
                    parts.append(.object(["type": .string("input_text"), "text": .string(text)]))
                case let .imageURL(url, _):
                    parts.append(.object(["type": .string("input_image"), "image_url": .string(url)]))
                case let .data(mediaType, data, _), let .file(mediaType, data, _, _):
                    guard topLevelMediaType(mediaType) == "image" else {
                        throw AIError.invalidArgument(argument: "messages", message: "Perplexity does not support file part media type \(mediaType).")
                    }
                    let resolved = isFullMediaType(mediaType) ? mediaType : try resolveFullMediaType(mediaType: mediaType, data: data)
                    parts.append(.object([
                        "type": .string("input_image"),
                        "image_url": .string("data:\(resolved);base64,\(data.base64EncodedString())")
                    ]))
                case .reasoning:
                    warnings.append(AIWarning(type: "unsupported", feature: "reasoning content in prompt"))
                case .providerReference:
                    throw AIError.invalidArgument(argument: "messages", message: "Perplexity does not support file parts with provider references.")
                case .reasoningFile, .custom, .toolCall, .toolResult, .toolApprovalRequest, .toolApprovalResponse:
                    throw AIError.invalidArgument(argument: "messages", message: "Perplexity does not support this user content part.")
                }
            }
            let textOnly = parts.allSatisfy { $0["type"]?.stringValue == "input_text" }
            input.append(.object([
                "type": .string("message"),
                "role": .string("user"),
                "content": textOnly
                    ? .string(parts.compactMap { $0["text"]?.stringValue }.joined())
                    : .array(parts)
            ]))
        case .assistant:
            let text = message.content.compactMap(\.text).joined()
            if !text.isEmpty {
                input.append(.object(["type": .string("message"), "role": .string("assistant"), "content": .string(text)]))
            }
            for part in message.content {
                switch part {
                case let .toolCall(call):
                    var value: [String: JSONValue] = [
                        "type": .string("function_call"), "call_id": .string(call.id),
                        "name": .string(call.name), "arguments": .string(call.arguments)
                    ]
                    if let signature = perplexityAgentThoughtSignature(call.providerMetadata) { value["thought_signature"] = .string(signature) }
                    input.append(.object(value))
                case let .toolResult(result):
                    input.append(try perplexityAgentToolResultInput(result))
                case .reasoning:
                    warnings.append(AIWarning(type: "unsupported", feature: "reasoning content in prompt"))
                case .text:
                    break
                case .imageURL, .data, .file, .reasoningFile, .custom, .providerReference, .toolApprovalRequest, .toolApprovalResponse:
                    throw AIError.invalidArgument(argument: "messages", message: "Perplexity does not support this assistant content part.")
                }
            }
        case .tool:
            for part in message.content {
                guard case let .toolResult(result) = part else {
                    throw AIError.invalidArgument(argument: "messages", message: "Perplexity does not support tool approval responses.")
                }
                input.append(try perplexityAgentToolResultInput(result))
            }
        }
    }
    return (input, warnings)
}

private func perplexityAgentToolResultInput(_ result: AIToolResult) throws -> JSONValue {
    var value: [String: JSONValue] = [
        "type": .string("function_call_output"),
        "call_id": .string(result.toolCallID),
        "name": .string(result.toolName),
        "output": .string(try perplexityAgentToolResultOutput(result))
    ]
    if let signature = perplexityAgentThoughtSignature(result.providerMetadata) { value["thought_signature"] = .string(signature) }
    return .object(value)
}

private func perplexityAgentThoughtSignature(_ metadata: [String: JSONValue]) -> String? {
    metadata["perplexity"]?["thoughtSignature"]?.stringValue
}

private func perplexityAgentToolResultOutput(_ result: AIToolResult) throws -> String {
    let output = result.modelOutput ?? result.result
    if let object = output.objectValue, let type = object["type"]?.stringValue {
        switch type {
        case "text", "error-text": return object["value"]?.stringValue ?? ""
        case "json", "error-json":
            return String(data: try encodeJSONBody(object["value"] ?? .null), encoding: .utf8) ?? "null"
        case "execution-denied": return object["reason"]?.stringValue ?? "Tool call execution denied."
        case "content":
            let values = object["value"]?.arrayValue ?? []
            guard values.allSatisfy({ $0["type"]?.stringValue == "text" }) else {
                throw AIError.invalidArgument(argument: "messages", message: "Perplexity does not support file and custom tool result content.")
            }
            return values.compactMap { $0["text"]?.stringValue }.joined()
        default: break
        }
    }
    return String(data: try encodeJSONBody(output), encoding: .utf8) ?? "null"
}

private func validatePerplexityAgentResponse(_ raw: JSONValue, providerID: String) throws {
    guard let output = raw["output"]?.arrayValue,
          raw["id"]?.stringValue != nil,
          raw["created_at"]?.doubleValue != nil,
          raw["model"]?.stringValue != nil,
          raw["object"]?.stringValue == "response",
          raw["status"]?.stringValue != nil else {
        throw perplexityAgentValidationError(providerID, "Perplexity Agent response envelope is invalid.")
    }
    for item in output {
        try validatePerplexityAgentOutputItem(item, providerID: providerID)
    }
    if let incomplete = raw["incomplete_details"], incomplete != .null {
        guard incomplete.objectValue != nil, incomplete["reason"]?.stringValue != nil else {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent incomplete_details is invalid.")
        }
    }
    if let error = raw["error"], error != .null {
        try validatePerplexityAgentError(error, providerID: providerID)
    }
    if let usage = raw["usage"], usage != .null {
        try validatePerplexityAgentUsage(usage, providerID: providerID)
    }
}

private func validatePerplexityAgentChunk(_ raw: JSONValue, providerID: String) throws {
    guard raw["type"]?.stringValue != nil else {
        throw perplexityAgentValidationError(providerID, "Perplexity Agent stream chunk type is invalid.")
    }
    for key in ["sequence_number", "output_index", "content_index"] {
        if let value = raw[key], value != .null, value.doubleValue == nil {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent stream chunk \(key) is invalid.")
        }
    }
    for key in ["item_id", "delta", "text", "thought"] {
        if let value = raw[key], value != .null, value.stringValue == nil {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent stream chunk \(key) is invalid.")
        }
    }
    for key in ["queries", "urls"] {
        if let value = raw[key], value != .null, !perplexityStringArray(value) {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent stream chunk \(key) is invalid.")
        }
    }
    if let response = raw["response"], response != .null { try validatePerplexityAgentResponse(response, providerID: providerID) }
    if let item = raw["item"], item != .null { try validatePerplexityAgentOutputItem(item, providerID: providerID) }
    if let results = raw["results"], results != .null {
        guard let results = results.arrayValue else {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent stream search results are invalid.")
        }
        for result in results { try validatePerplexityAgentSearchResult(result, providerID: providerID) }
    }
    if let contents = raw["contents"], contents != .null {
        guard let contents = contents.arrayValue else {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent stream fetched contents are invalid.")
        }
        for content in contents { try validatePerplexityAgentFetchedContent(content, providerID: providerID) }
    }
    if let error = raw["error"], error != .null { try validatePerplexityAgentError(error, providerID: providerID) }
}

private func validatePerplexityAgentOutputItem(_ item: JSONValue, providerID: String) throws {
    guard let type = item["type"]?.stringValue else {
        throw perplexityAgentValidationError(providerID, "Perplexity Agent output item type is invalid.")
    }
    switch type {
    case "message":
        guard perplexityNullableString(item["id"]) else {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent message id is invalid.")
        }
        if let content = item["content"], content != .null {
            guard let parts = content.arrayValue else {
                throw perplexityAgentValidationError(providerID, "Perplexity Agent message content is invalid.")
            }
            for part in parts {
                guard part.objectValue != nil,
                      part["type"]?.stringValue != nil,
                      perplexityOptionalString(part["text"]) else {
                    throw perplexityAgentValidationError(providerID, "Perplexity Agent message content part is invalid.")
                }
                if let annotations = part["annotations"], annotations != .null {
                    guard let annotations = annotations.arrayValue else {
                        throw perplexityAgentValidationError(providerID, "Perplexity Agent annotations are invalid.")
                    }
                    for annotation in annotations {
                        guard annotation.objectValue != nil,
                              perplexityOptionalString(annotation["type"]),
                              perplexityOptionalString(annotation["url"]),
                              perplexityOptionalString(annotation["title"]) else {
                            throw perplexityAgentValidationError(providerID, "Perplexity Agent annotation is invalid.")
                        }
                    }
                }
            }
        }
    case "search_results":
        if let results = item["results"], results != .null {
            guard let results = results.arrayValue else {
                throw perplexityAgentValidationError(providerID, "Perplexity Agent search results are invalid.")
            }
            for result in results { try validatePerplexityAgentSearchResult(result, providerID: providerID) }
        }
    case "fetch_url_results":
        if let contents = item["contents"], contents != .null {
            guard let contents = contents.arrayValue else {
                throw perplexityAgentValidationError(providerID, "Perplexity Agent fetched contents are invalid.")
            }
            for content in contents { try validatePerplexityAgentFetchedContent(content, providerID: providerID) }
        }
    case "function_call":
        for key in ["id", "call_id", "name", "arguments", "thought_signature"] where !perplexityNullableString(item[key]) {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent function call \(key) is invalid.")
        }
    default:
        break
    }
}

private func validatePerplexityAgentSearchResult(_ result: JSONValue, providerID: String) throws {
    guard result.objectValue != nil,
          result["title"]?.stringValue != nil,
          result["url"]?.stringValue != nil,
          perplexityOptionalNumber(result["id"]),
          perplexityOptionalString(result["snippet"]),
          perplexityNullableString(result["date"]),
          perplexityNullableString(result["last_updated"]),
          perplexityOptionalString(result["source"]) else {
        throw perplexityAgentValidationError(providerID, "Perplexity Agent search result is invalid.")
    }
}

private func validatePerplexityAgentFetchedContent(_ content: JSONValue, providerID: String) throws {
    guard content.objectValue != nil,
          content["title"]?.stringValue != nil,
          content["url"]?.stringValue != nil,
          perplexityOptionalString(content["snippet"]) else {
        throw perplexityAgentValidationError(providerID, "Perplexity Agent fetched content is invalid.")
    }
}

private func validatePerplexityAgentError(_ error: JSONValue, providerID: String) throws {
    guard error.objectValue != nil,
          error["message"]?.stringValue != nil,
          perplexityNullableString(error["code"]),
          perplexityNullableString(error["type"]) else {
        throw perplexityAgentValidationError(providerID, "Perplexity Agent error is invalid.")
    }
}

private func validatePerplexityAgentUsage(_ usage: JSONValue, providerID: String) throws {
    guard usage.objectValue != nil,
          usage["input_tokens"]?.doubleValue != nil,
          usage["output_tokens"]?.doubleValue != nil,
          usage["total_tokens"]?.doubleValue != nil else {
        throw perplexityAgentValidationError(providerID, "Perplexity Agent usage is invalid.")
    }
    for (containerKey, keys) in [
        ("input_tokens_details", ["cached_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]),
        ("output_tokens_details", ["reasoning_tokens"])
    ] {
        if let container = usage[containerKey], container != .null {
            guard let object = container.objectValue, keys.allSatisfy({ perplexityOptionalNumber(object[$0]) }) else {
                throw perplexityAgentValidationError(providerID, "Perplexity Agent \(containerKey) is invalid.")
            }
        }
    }
    if let details = usage["tool_calls_details"], details != .null {
        guard let details = details.objectValue else {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent tool_calls_details is invalid.")
        }
        for detail in details.values {
            guard let detail = detail.objectValue, perplexityOptionalNumber(detail["invocation"]) else {
                throw perplexityAgentValidationError(providerID, "Perplexity Agent tool call usage is invalid.")
            }
        }
    }
    if let cost = usage["cost"], cost != .null {
        guard let cost = cost.objectValue,
              perplexityOptionalString(cost["currency"]),
              ["input_cost", "output_cost", "total_cost"].allSatisfy({ perplexityOptionalNumber(cost[$0]) }),
              ["cache_creation_cost", "cache_read_cost", "tool_calls_cost"].allSatisfy({ perplexityNullableNumber(cost[$0]) }) else {
            throw perplexityAgentValidationError(providerID, "Perplexity Agent cost is invalid.")
        }
    }
}

private func perplexityOptionalString(_ value: JSONValue?) -> Bool { value == nil || value?.stringValue != nil }
private func perplexityNullableString(_ value: JSONValue?) -> Bool { value == nil || value == .null || value?.stringValue != nil }
private func perplexityOptionalNumber(_ value: JSONValue?) -> Bool { value == nil || value?.doubleValue != nil }
private func perplexityNullableNumber(_ value: JSONValue?) -> Bool { value == nil || value == .null || value?.doubleValue != nil }

private func perplexityAgentValidationError(_ providerID: String, _ message: String) -> AIError {
    .invalidResponse(provider: providerID, message: message)
}

private struct PerplexityPreparedCall {
    var body: [String: JSONValue]
    var warnings: [AIWarning]
}

private func perplexityPreparedCall(for request: LanguageModelRequest, modelID: String, stream: Bool) throws -> PerplexityPreparedCall {
    var options = perplexityProviderOptions(from: request)
    let responseFormat = perplexityResolvedResponseFormat(request: request, options: &options)
    var body: [String: JSONValue] = [
        "model": .string(modelID),
        "messages": .array(try request.messages.map(perplexityMessageJSON))
    ]
    if stream { body["stream"] = true }
    if let temperature = request.temperature { body["temperature"] = .number(temperature) }
    if let topP = request.topP { body["top_p"] = .number(topP) }
    if let topK = request.topK { body["top_k"] = .number(Double(topK)) }
    if let presencePenalty = request.presencePenalty { body["presence_penalty"] = .number(presencePenalty) }
    if let frequencyPenalty = request.frequencyPenalty { body["frequency_penalty"] = .number(frequencyPenalty) }
    if let maxOutputTokens = request.maxOutputTokens { body["max_tokens"] = .number(Double(maxOutputTokens)) }
    if let responseFormat = perplexityResponseFormat(from: responseFormat) {
        body["response_format"] = responseFormat
    }
    body.merge(options) { _, new in new }
    return PerplexityPreparedCall(body: body, warnings: perplexityWarnings(for: request))
}

private func perplexityProviderOptions(from request: LanguageModelRequest) -> [String: JSONValue] {
    var output = request.extraBody
    if let nested = output.removeValue(forKey: "perplexity")?.objectValue {
        output.merge(nested) { _, nested in nested }
    }
    if let nested = request.providerOptions["perplexity"]?.objectValue {
        output.merge(nested) { _, nested in nested }
    }
    return output
}

private func perplexityResolvedResponseFormat(request: LanguageModelRequest, options: inout [String: JSONValue]) -> JSONValue? {
    if let responseFormat = request.responseFormat {
        options.removeValue(forKey: "responseFormat")
        return perplexityResponseFormatJSON(responseFormat)
    }
    return options.removeValue(forKey: "responseFormat")
}

private func perplexityResponseFormatJSON(_ responseFormat: AIResponseFormat) -> JSONValue? {
    switch responseFormat {
    case .text:
        return nil
    case let .json(schema, name, description):
        return .object([
            "type": .string("json"),
            "schema": schema,
            "name": name.map(JSONValue.string),
            "description": description.map(JSONValue.string)
        ])
    }
}

private func perplexityResponseFormat(from value: JSONValue?) -> JSONValue? {
    guard value?["type"]?.stringValue == "json" else { return nil }
    var jsonSchema: [String: JSONValue] = [:]
    if let schema = value?["schema"] {
        jsonSchema["schema"] = schema
    }
    return .object([
        "type": .string("json_schema"),
        "json_schema": .object(jsonSchema)
    ])
}

private func perplexityWarnings(for request: LanguageModelRequest) -> [AIWarning] {
    var warnings: [AIWarning] = []
    if request.topK != nil {
        warnings.append(AIWarning(type: "unsupported", feature: "topK"))
    }
    if !request.stopSequences.isEmpty {
        warnings.append(AIWarning(type: "unsupported", feature: "stopSequences"))
    }
    if request.seed != nil {
        warnings.append(AIWarning(type: "unsupported", feature: "seed"))
    }
    if isCustomReasoning(request.reasoning) {
        warnings.append(AIWarning(
            type: "unsupported",
            feature: "reasoning",
            message: "This provider does not support reasoning configuration."
        ))
    }
    return warnings
}

private func perplexitySources(from citations: JSONValue?) -> [AISource] {
    citations?.arrayValue?.enumerated().compactMap { index, citation in
        guard let url = citation.stringValue else { return nil }
        return AISource(
            id: "citation-\(index)",
            sourceType: "url",
            url: url,
            providerMetadata: ["perplexity": .object(["citationIndex": .number(Double(index))])],
            rawValue: citation
        )
    } ?? []
}

private func perplexityProviderMetadata(from raw: JSONValue) -> [String: JSONValue] {
    [
        "perplexity": .object([
            "usage": perplexityUsageMetadata(from: raw["usage"]),
            "cost": perplexityCostMetadata(from: raw["usage"]?["cost"]),
            "images": perplexityImagesMetadata(from: raw["images"])
        ])
    ]
}

private func perplexityEmptyProviderMetadata() -> [String: JSONValue] {
    [
        "perplexity": .object([
            "usage": .object([
                "citationTokens": .null,
                "numSearchQueries": .null
            ]),
            "cost": .null,
            "images": .null
        ])
    ]
}

private func perplexityMergeProviderMetadata(from raw: JSONValue, into metadata: inout [String: JSONValue]) {
    var perplexity = metadata["perplexity"]?.objectValue
        ?? perplexityEmptyProviderMetadata()["perplexity"]?.objectValue
        ?? [:]
    if let usage = raw["usage"] {
        perplexity["usage"] = perplexityUsageMetadata(from: usage)
        perplexity["cost"] = perplexityCostMetadata(from: usage["cost"])
    }
    if raw["images"] != nil {
        perplexity["images"] = perplexityImagesMetadata(from: raw["images"])
    }
    metadata["perplexity"] = .object(perplexity)
}

private func perplexityUsageMetadata(from value: JSONValue?) -> JSONValue {
    .object([
        "citationTokens": value?["citation_tokens"] ?? .null,
        "numSearchQueries": value?["num_search_queries"] ?? .null
    ])
}

private func perplexityCostMetadata(from value: JSONValue?) -> JSONValue {
    guard let value else { return .null }
    return .object([
        "inputTokensCost": value["input_tokens_cost"] ?? .null,
        "outputTokensCost": value["output_tokens_cost"] ?? .null,
        "requestCost": value["request_cost"] ?? .null,
        "totalCost": value["total_cost"] ?? .null
    ])
}

private func perplexityImagesMetadata(from value: JSONValue?) -> JSONValue {
    guard let images = value?.arrayValue else { return .null }
    return .array(images.map { image in
        .object([
            "imageUrl": image["image_url"] ?? .null,
            "originUrl": image["origin_url"] ?? .null,
            "height": image["height"] ?? .null,
            "width": image["width"] ?? .null
        ])
    })
}

private func perplexityUsage(from raw: JSONValue) -> TokenUsage? {
    guard let usage = raw["usage"] else { return nil }
    let inputTokens = usage["prompt_tokens"]?.intValue ?? 0
    let outputTokens = usage["completion_tokens"]?.intValue ?? 0
    let reasoningTokens = usage["reasoning_tokens"]?.intValue ?? 0
    return TokenUsage(
        inputTokens: inputTokens,
        outputTokens: outputTokens + reasoningTokens,
        totalTokens: inputTokens + outputTokens + reasoningTokens,
        inputTokensNoCache: inputTokens,
        outputTextTokens: outputTokens,
        outputReasoningTokens: reasoningTokens,
        rawValue: usage
    )
}

private func perplexityAgentAppendOutputItem(
    _ item: JSONValue,
    content: inout [AIResultContentPart],
    sourceIndexes: inout [String: Int],
    hasFunctionCall: inout Bool
) {
    switch item["type"]?.stringValue {
    case "message":
        for part in item["content"]?.arrayValue ?? [] {
            if part["type"]?.stringValue == "output_text", let text = part["text"]?.stringValue {
                content.append(.text(text))
            }
            for annotation in part["annotations"]?.arrayValue ?? [] {
                if let source = perplexityAgentAnnotationSource(annotation) {
                    perplexityAgentAddSource(source, content: &content, sourceIndexes: &sourceIndexes)
                }
            }
        }
    case "search_results":
        for result in item["results"]?.arrayValue ?? [] {
            if let source = perplexityAgentSearchSource(result) {
                perplexityAgentAddSource(source, content: &content, sourceIndexes: &sourceIndexes)
            }
        }
    case "fetch_url_results":
        for result in item["contents"]?.arrayValue ?? [] {
            if let source = perplexityAgentFetchedSource(result) {
                perplexityAgentAddSource(source, content: &content, sourceIndexes: &sourceIndexes)
            }
        }
    case "function_call":
        guard let callID = item["call_id"]?.stringValue,
              let name = item["name"]?.stringValue,
              let arguments = item["arguments"]?.stringValue else { return }
        hasFunctionCall = true
        content.append(.toolCall(perplexityAgentToolCall(from: item, callID: callID, name: name, arguments: arguments)))
    default:
        break
    }
}

private func perplexityAgentAddSource(_ source: AISource, content: inout [AIResultContentPart], sourceIndexes: inout [String: Int]) {
    guard let url = source.url else { return }
    if let index = sourceIndexes[url] {
        if perplexityAgentSourceHasResultID(source),
           case let .source(existing) = content[index],
           !perplexityAgentSourceHasResultID(existing) {
            content[index] = .source(source)
        }
        return
    }
    sourceIndexes[url] = content.count
    content.append(.source(source))
}

private func perplexityAgentSearchSource(_ result: JSONValue) -> AISource? {
    guard let url = result["url"]?.stringValue else { return nil }
    let resultID = result["id"]?.intValue
    return AISource(
        id: resultID.map { String($0) } ?? generateId(),
        sourceType: "url",
        url: url,
        title: result["title"]?.stringValue,
        providerMetadata: ["perplexity": .object([
            "resultId": resultID.map { .number(Double($0)) } ?? .null,
            "snippet": result["snippet"] ?? .null,
            "date": result["date"] ?? .null,
            "lastUpdated": result["last_updated"] ?? .null,
            "source": result["source"] ?? .null
        ])],
        rawValue: result
    )
}

private func perplexityAgentAnnotationSource(_ annotation: JSONValue) -> AISource? {
    guard let url = annotation["url"]?.stringValue else { return nil }
    return AISource(id: generateId(), sourceType: "url", url: url, title: annotation["title"]?.stringValue, rawValue: annotation)
}

private func perplexityAgentFetchedSource(_ result: JSONValue) -> AISource? {
    guard let url = result["url"]?.stringValue else { return nil }
    return AISource(
        id: generateId(),
        sourceType: "url",
        url: url,
        title: result["title"]?.stringValue,
        providerMetadata: ["perplexity": .object(["snippet": result["snippet"] ?? .null])],
        rawValue: result
    )
}

private func perplexityAgentSourceHasResultID(_ source: AISource) -> Bool {
    source.providerMetadata["perplexity"]?["resultId"]?.intValue != nil
}

private func perplexityAgentToolCall(from item: JSONValue, callID: String, name: String, arguments: String) -> AIToolCall {
    var metadata: [String: JSONValue] = ["itemId": item["id"] ?? .null]
    if let signature = item["thought_signature"]?.stringValue { metadata["thoughtSignature"] = .string(signature) }
    return AIToolCall(
        id: callID,
        name: name,
        arguments: arguments,
        providerMetadata: ["perplexity": .object(metadata)],
        rawValue: item
    )
}

private func perplexityAgentFinishReason(status: String?, incompleteReason: String?, hasFunctionCall: Bool) -> String {
    switch incompleteReason {
    case "max_output_tokens": return "length"
    case "content_filter": return "content-filter"
    default: break
    }
    switch status {
    case "completed": return hasFunctionCall ? "tool-calls" : "stop"
    case "requires_action": return "tool-calls"
    case "failed": return "error"
    default: return hasFunctionCall ? "tool-calls" : "other"
    }
}

private func perplexityAgentUsage(from value: JSONValue?) -> TokenUsage? {
    guard let value, value != .null else { return nil }
    let input = value["input_tokens"]?.intValue ?? 0
    let output = value["output_tokens"]?.intValue ?? 0
    let cacheRead = value["input_tokens_details"]?["cache_read_input_tokens"]?.intValue
        ?? value["input_tokens_details"]?["cached_tokens"]?.intValue ?? 0
    let cacheWrite = value["input_tokens_details"]?["cache_creation_input_tokens"]?.intValue ?? 0
    let reasoning = value["output_tokens_details"]?["reasoning_tokens"]?.intValue ?? 0
    return TokenUsage(
        inputTokens: input,
        outputTokens: output,
        totalTokens: value["total_tokens"]?.intValue ?? input + output,
        inputTokensNoCache: max(0, input - cacheRead - cacheWrite),
        inputTokensCacheRead: cacheRead,
        inputTokensCacheWrite: cacheWrite,
        outputTextTokens: max(0, output - reasoning),
        outputReasoningTokens: reasoning,
        rawValue: value
    )
}

private func perplexityAgentProviderMetadata(from usage: JSONValue?) -> [String: JSONValue] {
    let details = usage?["tool_calls_details"]?.objectValue ?? [:]
    let searchCalls = details.reduce(0) { total, entry in
        entry.key.contains("search") ? total + (entry.value["invocation"]?.intValue ?? 0) : total
    }
    let toolCalls: [String: JSONValue] = details.mapValues { detail in
        JSONValue.object(["invocation": detail["invocation"] ?? .null])
    }
    let cost = usage?["cost"]
    let costMetadata: JSONValue
    if let cost, cost != .null {
        let fields: [String: JSONValue] = [
            "inputTokensCost": cost["input_cost"] ?? .null,
            "outputTokensCost": cost["output_cost"] ?? .null,
            "requestCost": .null,
            "totalCost": cost["total_cost"] ?? .null,
            "currency": cost["currency"] ?? .null,
            "cacheCreationCost": cost["cache_creation_cost"] ?? .null,
            "cacheReadCost": cost["cache_read_cost"] ?? .null,
            "toolCallsCost": cost["tool_calls_cost"] ?? .null
        ]
        costMetadata = .object(fields)
    } else {
        costMetadata = .null
    }
    let metadata: [String: JSONValue] = [
        "usage": .object(["citationTokens": .null, "numSearchQueries": details.isEmpty ? .null : .number(Double(searchCalls))]),
        "images": .null,
        "cost": costMetadata,
        "toolCalls": details.isEmpty ? .null : .object(toolCalls)
    ]
    return ["perplexity": .object(metadata)]
}

private func perplexityAgentResponseMetadata(from raw: JSONValue, response: AIHTTPResponse, modelID: String) -> AIResponseMetadata {
    AIResponseMetadata(
        id: raw["id"]?.stringValue,
        timestamp: raw["created_at"]?.doubleValue.map { Date(timeIntervalSince1970: $0) },
        modelID: raw["model"]?.stringValue ?? modelID,
        headers: response.headers,
        body: raw
    )
}

private func perplexityFinishReason(_ reason: String?) -> String? {
    switch reason {
    case "stop", "length":
        return reason
    default:
        return "other"
    }
}

private func validatePerplexityGenerateResponse(_ raw: JSONValue, providerID: String) throws {
    guard raw["id"]?.stringValue != nil,
          raw["created"]?.doubleValue != nil,
          raw["model"]?.stringValue != nil,
          let choices = raw["choices"]?.arrayValue else {
        throw AIError.invalidResponse(provider: providerID, message: "Perplexity response is invalid.")
    }
    for choice in choices {
        guard choice["message"]?["role"]?.stringValue == "assistant",
              choice["message"]?["content"]?.stringValue != nil else {
            throw AIError.invalidResponse(provider: providerID, message: "Perplexity response is invalid.")
        }
        if let finishReason = choice["finish_reason"], finishReason != .null, finishReason.stringValue == nil {
            throw AIError.invalidResponse(provider: providerID, message: "Perplexity response is invalid.")
        }
    }
    if let citations = raw["citations"], citations != .null {
        guard citations.arrayValue?.allSatisfy({ $0.stringValue != nil }) == true else {
            throw AIError.invalidResponse(provider: providerID, message: "Perplexity response is invalid.")
        }
    }
    if let images = raw["images"], images != .null {
        guard let array = images.arrayValue else {
            throw AIError.invalidResponse(provider: providerID, message: "Perplexity response is invalid.")
        }
        for image in array {
            guard image["image_url"]?.stringValue != nil,
                  image["origin_url"]?.stringValue != nil,
                  image["height"]?.doubleValue != nil,
                  image["width"]?.doubleValue != nil else {
                throw AIError.invalidResponse(provider: providerID, message: "Perplexity response is invalid.")
            }
        }
    }
    if let usage = raw["usage"], usage != .null {
        guard usage["prompt_tokens"]?.doubleValue != nil,
              usage["completion_tokens"]?.doubleValue != nil else {
            throw AIError.invalidResponse(provider: providerID, message: "Perplexity response is invalid.")
        }
    }
}

private func perplexityMessageJSON(_ message: AIMessage) throws -> JSONValue {
    guard message.role != .tool else {
        throw AIError.invalidArgument(argument: "messages", message: "Perplexity does not support tool messages.")
    }

    let multipart = message.content.contains { part in
        switch part {
        case .text, .reasoning:
            return false
        case .imageURL:
            return true
        case let .data(mimeType, _, _), let .file(mimeType, _, _, _):
            return mimeType.hasPrefix("image/") || mimeType == "application/pdf"
        case .reasoningFile, .custom, .providerReference, .toolCall, .toolResult, .toolApprovalRequest, .toolApprovalResponse:
            return false
        }
    }

    guard multipart else {
        return .object([
            "role": .string(message.role.rawValue),
            "content": .string(message.content.compactMap(\.text).joined())
        ])
    }

    let parts = message.content.enumerated().compactMap { index, part -> JSONValue? in
        switch part {
        case let .text(text, _):
            return .object(["type": .string("text"), "text": .string(text)])
        case let .reasoning(text, _):
            return .object(["type": .string("text"), "text": .string(text)])
        case let .imageURL(url, _):
            return .object([
                "type": .string("image_url"),
                "image_url": .object(["url": .string(url)])
            ])
        case let .data(mimeType, data, _) where mimeType == "application/pdf":
            return .object([
                "type": .string("file_url"),
                "file_url": .object(["url": .string(data.base64EncodedString())]),
                "file_name": .string("document-\(index).pdf")
            ])
        case let .file(mimeType, data, filename, _) where mimeType == "application/pdf":
            return .object([
                "type": .string("file_url"),
                "file_url": .object(["url": .string(data.base64EncodedString())]),
                "file_name": .string(filename ?? "document-\(index).pdf")
            ])
        case let .data(mimeType, data, _) where mimeType.hasPrefix("image/"),
             let .file(mimeType, data, _, _) where mimeType.hasPrefix("image/"):
            return .object([
                "type": .string("image_url"),
                "image_url": .object(["url": .string("data:\(mimeType);base64,\(data.base64EncodedString())")])
            ])
        case .data, .file, .reasoningFile, .custom, .providerReference, .toolCall, .toolResult, .toolApprovalRequest, .toolApprovalResponse:
            return nil
        }
    }

    return .object([
        "role": .string(message.role.rawValue),
        "content": .array(parts)
    ])
}