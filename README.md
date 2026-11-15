<!-- Generated distribution; edit the canonical repository. -->
# swift-ai-perplexity 1.2.1

Independent Swift-native port of Vercel AI SDK. Source and upstream dispositions are in PROVENANCE.json and Porting/.

## Distribution manifest

Add this package to your SwiftPM dependencies:

```swift
.package(url: "https://github.com/SwiftAISDK/swift-ai-perplexity.git", from: "1.2.1")
```

Products: `SwiftAIPerplexity`.
Contribute and report issues in the canonical SwiftAISDK repository.

## Authored usage guide

# SwiftAIPerplexity

Swift-native port of `@ai-sdk/perplexity`. Maintained in the canonical SwiftAISDK repository and exported as an independent Swift package.

The canonical repository owns all source edits. This package owns its provider
configuration and wire behavior, using only the declared shared dependencies.
Existing upstream baselines, capabilities and documented gaps remain unchanged.
See `Porting/map.json` and the canonical provider guide.

See [provider usage and capabilities](https://github.com/SwiftAISDK/SwiftAISDK/blob/main/docs-site/scripts/provider-guides/perplexity.md) for authentication, options and supported surfaces.
## Behavior notes

Agent requests with `toolChoice: "none"` omit native and function tool definitions and set `max_tool_calls` to zero, including presets. Conversation tool-call history is preserved.

## Current upstream review

The reviewed baseline is `@ai-sdk/perplexity` 5.0.34. Native decisions accept `PerplexityDecisionModelOptions(mode: .webGrounding, maxStaleness: 0)`. `mode` sends `web_grounding`; maximum staleness is a nonnegative integer representing fetched-page age in seconds, including zero.

Results expose `searchResults`, `fetchedResults` and exact `toolCalls` invocation counts under `providerMetadata.perplexity`. Missing invocation counts remain null and reported zeros are preserved. HTTP failures retain grounding-count headers and retryability. Unknown decision options still produce unsupported warnings; invalid grounding options fail before HTTP.
