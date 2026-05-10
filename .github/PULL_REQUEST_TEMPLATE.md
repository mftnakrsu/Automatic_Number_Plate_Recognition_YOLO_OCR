## Summary

<!-- 1-3 sentences: what does this PR change, and why? -->

## Type

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behavior change)
- [ ] Docs
- [ ] CI / build
- [ ] Test only

## Test plan

<!-- How did you verify this works? -->

- [ ] `make lint` clean
- [ ] `make type` clean
- [ ] `make test` green
- [ ] (if API changed) manual smoke against `anpr serve` / `/api/v1/infer`

## Related

<!-- Issues, prior PRs, RFCs -->

## Privacy / KVKK check

- [ ] No code path persists raw plate text (only `hash_plate(...)` output).
- [ ] N/A — this PR doesn't touch storage.
