# Changelog

## 1.2.0

- Add typed web-grounded Decision mode and maximum-staleness controls. Preserve fetched/search results, exact tool invocation counts, grounding error headers and reported zero values. Reviewed npm delta: @ai-sdk/perplexity 5.0.28 to 5.0.34. Validation: `node Scripts/test-packages.mjs` passed all 52 owning suites and 1731 facade tests; tooling 27/27, generation/isolation checks, standalone exports and 106 minimum/latest consumers, documentation check and 101-page build passed. Publication requires green exact-source Swift 6.3.3 CI. Paid/live verification and full-package parity are not claimed.
  Compatibility: Additive native APIs or capabilities under the project minor-release convention. npm baselines and Swift SemVer are independent; this reviewed delta does not establish full-package or live-service parity.
