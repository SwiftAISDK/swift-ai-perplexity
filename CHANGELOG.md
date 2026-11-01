# Changelog

## 1.1.0

- Expose first-seen deduplicated searchQueries from generation search-result items and streaming reasoning/output/terminal-response items; no reported queries returns JSON null; fetch URLs remain distinct. Validate nullable search-results queries as string arrays, including malformed native response rejection. Reviewed upstream: @ai-sdk/perplexity 5.0.20 to 5.0.24. Validation passed: node Scripts/test-packages.mjs (all 52 owning package suites and the compatibility facade); node Scripts/sdk.mjs generate --check; node --test Scripts/tests/*.test.cjs (27 passed); docs-site check and build (100 pages). Live provider calls were not run.
  Compatibility: Minor release for additive native API; existing static settings and call sites remain compatible. Updated direct dependency minimums are recorded in the manifest. Existing documented JS/framework and native product gaps remain; version-delta review does not imply full parity or live-service testing. Explicit custom authentication skips an unused credential supplier, a deliberate native callback-order adaptation.
