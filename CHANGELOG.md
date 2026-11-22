# Changelog

## 1.3.1

- Align the provider request identity with @ai-sdk/perplexity 5.0.37 after comparing source, declarations and matching-release tests against 5.0.35. Offline validation passed: node Scripts/test-packages.mjs (56 packages, 3672 owning tests and 1760 compatibility tests), generation, tooling, isolation, and version selection. See Docs/Archive/WeeklyUpstreamAudit20261122.md for retained gaps. Paid provider calls were skipped.
  Compatibility: Source-compatible request-header update.
