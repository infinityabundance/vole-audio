# Receipts

Immutable evidence outputs.

- Each court/probe run writes one JSON receipt under
  `<court>/<court>-<runid>.json` with `create_new` semantics: an existing
  receipt is **never** overwritten or rewritten after code changes.
- Receipt schema: `vole.audio.evidence.v1` (see `docs/EVIDENCE.md`).
- Every receipt self-verifies: `receipt_sha256` binds the canonical JSON.
- Raw traces are written next to receipts when a court records them; the
  receipt stores the trace path, byte count, and SHA-256.

`git status` hygiene: receipt JSON files are evidence and are committed;
bulk trace/artifact payloads are gitignored (`receipts/traces/`,
`receipts/artifacts/`) and referenced by hash from the committed receipt.
