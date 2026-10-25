import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


/// Perplexity's native Decisions API with shared text, JSON and image evidence.
public final class PerplexityDecisionModel: AIDecisionModelV4, @unchecked Sendable {
    public let specificationVersion = "v4"
    public let supportedQuestionTypes: [AIDecisionQuestionType] = [.choice, .score, .boolean]
    public let providerID: String
    public let modelID: String
    private let config: PerplexityModelConfiguration

    public init(modelID: String, config: PerplexityModelConfiguration) {
        self.modelID = modelID
        self.providerID = config.providerID
        self.config = config
    }

    public func doDecide(_ options: AIDecisionModelV4CallOptions) async throws -> AIDecisionModelV4Result {
        guard (1...128).contains(options.questions.count), !options.questions.keys.contains("") else {
            throw AIError.invalidArgument(argument: "questions", message: "Perplexity Decisions requires 1 to 128 questions with non-empty names.")
        }
        var questions: [String: JSONValue] = [:]
        for (id, question) in options.questions {
            let type: String
            let criteria: JSONValue?
            switch question {
            case let .choice(_, values):
                guard (1...255).contains(values.count) else {
                    throw AIError.invalidArgument(argument: "questions.\(id).criteria", message: "Perplexity Choice questions support 1 to 255 options.")
                }
                type = "choice"
                criteria = .object(values)
            case let .score(_, values):
                guard (1...10).contains(values.count), !values.contains(.null) else {
                    throw AIError.invalidArgument(argument: "questions.\(id).criteria", message: "Perplexity Score questions require 1 to 10 non-null levels.")
                }
                type = "score"
                criteria = .array(values)
            case let .boolean(_, values):
                type = "noul"
                criteria = values.map(JSONValue.object)
            case .box:
                throw AIDecisionUnsupportedQuestionTypeError(questionID: id, questionType: question.type, providerID: providerID, modelID: modelID)
            }
            questions[id] = .object(["type": .string(type), "instructions": question.instructions, "criteria": criteria])
        }
        let state = try perplexityDecisionState(options.state)
        let result = try await config.sendJSONResponse(
            path: "/v1/decisions", modelID: modelID,
            body: .object(["model": .string(modelID), "state": state, "questions": .object(questions)]),
            headers: options.headers, abortSignal: options.abortSignal
        )
        let raw = result.json
        guard let nativeAnswers = raw["answers"]?.objectValue,
              Set(nativeAnswers.keys) == Set(options.questions.keys) else {
            throw invalid("Perplexity Decisions must return exactly one answer for every question.")
        }
        var answers: [String: AIDecisionAnswer] = [:]
        var confidence: [String: JSONValue] = [:]
        var legend: [String: JSONValue] = [:]
        for (id, answer) in nativeAnswers {
            switch answer["type"]?.stringValue {
            case "noul":
                answers[id] = .boolean(probability: try probability(answer["noul"]))
            case "choice", "score":
                guard let values = answer["probabilities"]?.objectValue else { throw invalid("Missing probability distribution.") }
                let probabilities = try values.mapValues { try probability($0) }
                if answer["type"] == "choice" {
                    guard let choice = answer["choice"]?.stringValue else { throw invalid("Missing choice.") }
                    answers[id] = .choice(choice: choice, probabilities: probabilities)
                } else {
                    guard let score = answer["score"]?.doubleValue, score.isFinite else { throw invalid("Invalid score.") }
                    answers[id] = .score(score: score, probabilities: probabilities)
                    if let value = answer["legend"], value != .null {
                        guard value.objectValue != nil, isFiniteJSON(value) else { throw invalid("Invalid score legend.") }
                        legend[id] = value
                    }
                }
                if let value = answer["confidence"], value != .null {
                    confidence[id] = .number(try probability(value))
                }
            default:
                throw invalid("Unsupported or malformed Perplexity decision answer.")
            }
        }
        var warnings: [AIWarning] = []
        if options.maxOutputTokens != nil { warnings.append(.init(type: "unsupported", feature: "maxOutputTokens")) }
        if options.reasoning != nil { warnings.append(.init(type: "unsupported", feature: "reasoning")) }
        warnings += (options.providerOptions["perplexity"]?.objectValue ?? [:]).keys.sorted().map {
            AIWarning(type: "unsupported", feature: "providerOptions.perplexity.\($0)")
        }
        var usage: AIDecisionModelUsage?
        if let value = raw["usage"], value != .null {
            guard let object = value.objectValue else { throw invalid("Invalid usage.") }
            func tokens(_ name: String) throws -> Int? {
                guard let value = object[name], value != .null else { return nil }
                guard let count = value.intValue else { throw invalid("Invalid usage token count.") }
                return count
            }
            usage = try .init(inputTokens: tokens("input_tokens"), outputTokens: tokens("output_tokens"))
        }
        if let model = raw["model"], model != .null, model.stringValue == nil { throw invalid("Invalid response model.") }
        return AIDecisionModelV4Result(
            answers: answers, usage: usage, warnings: warnings,
            providerMetadata: ["perplexity": .object(["confidence": .object(confidence), "legend": .object(legend)])],
            response: .init(id: result.response.headerValue("x-request-id"), modelID: raw["model"]?.stringValue ?? modelID,
                            headers: result.response.headers, body: raw)
        )
    }

    private func probability(_ value: JSONValue?) throws -> Double {
        guard let value = value?.doubleValue, value.isFinite, (0...1).contains(value) else { throw invalid("Invalid probability.") }
        return value
    }

    private func invalid(_ message: String) -> AIError { .invalidResponse(provider: providerID, message: message) }
}

private func perplexityDecisionState(_ state: [AIDecisionStatePart]) throws -> JSONValue {
    let parts: [JSONValue] = try state.map { part in
        switch part {
        case let .text(text): return .string(text)
        case let .json(value): return value
        case let .file(mediaType, data, _, _):
            let bytes: Data
            let base64: String
            switch data {
            case let .data(value): bytes = value; base64 = value.base64EncodedString()
            case let .base64(value):
                guard let decoded = Data(base64Encoded: value) else { throw AIError.invalidArgument(argument: "state", message: "File data must be valid base64.") }
                bytes = decoded; base64 = value
            case .url, .reference:
                throw AIDecisionUnsupportedFunctionalityError(functionality: "Perplexity decision file input requires data")
            }
            let resolved = try resolveFullMediaType(mediaType: mediaType, data: bytes)
            guard ["image/png", "image/jpeg", "image/webp"].contains(resolved) else {
                throw AIDecisionUnsupportedFunctionalityError(functionality: "Perplexity decision image media type: \(mediaType)")
            }
            return .object(["type": "image_url", "image_url": ["url": .string("data:\(resolved);base64,\(base64)")]])
        }
    }
    guard try countPerplexityDecisionImages(.array(parts)) <= 8 else {
        throw AIError.invalidArgument(argument: "state", message: "Perplexity Decisions supports at most 8 images in the shared state.")
    }
    if parts.count == 1, let first = parts.first {
        switch first {
        case .string, .object, .array: return first
        default: break
        }
    }
    return .array(parts)
}

private func countPerplexityDecisionImages(_ value: JSONValue) throws -> Int {
    switch value {
    case let .array(values): return try values.reduce(0) { try $0 + countPerplexityDecisionImages($1) }
    case let .object(object):
        if object["type"] == "image_url" {
            guard let url = object["image_url"]?["url"]?.stringValue?.lowercased(),
                  ["data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,"].contains(where: url.hasPrefix) else {
                throw AIDecisionUnsupportedFunctionalityError(functionality: "Perplexity decision images require PNG, JPEG, or WebP base64 data URLs")
            }
            return 1
        }
        return try object.values.reduce(0) { try $0 + countPerplexityDecisionImages($1) }
    default: return 0
    }
}