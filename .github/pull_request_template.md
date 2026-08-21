## Summary

<!-- What changed? Keep this concrete and scoped. -->

## Motivation

<!-- Why is this change needed? -->

## Validation

<!-- List exact commands and results. Do not claim checks that were not run. -->

## Risk and rollback

<!-- Compatibility impact, operational risk, and how to revert safely. -->

## Checklist

- [ ] The PR has one primary responsibility and no unrelated cleanup.
- [ ] Tests cover changed behavior, or the reason they do not is documented.
- [ ] The change serves the current pretraining or AnyRes SFT/GRPO lineage.
- [ ] Official release-lock, model/config/tokenizer compatibility changes are documented.
- [ ] No corpus, JSONL/CSV/TSV, media, model weight, checkpoint, credential, or log is included.
- [ ] Any new script or tokenizer tool has a production consumer and explicit hygiene allowlist entry.
- [ ] Documentation and runbooks are updated when behavior or operations change.
