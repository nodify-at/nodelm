# Quality gates

A meaningful change is complete only after all applicable gates below pass.

1. Understand affected contracts, callers, tests, boundaries, errors, and compatibility.
2. Keep the patch to the smallest coherent scope; no unrelated refactors or placeholders.
3. Preserve dependency direction and avoid infrastructure leakage into domain contracts.
4. Keep typed Python strict: no silent `Any` escape, ignored errors, or weakened checks.
5. Review async/process lifecycle, timeouts, cleanup, retries, idempotency, and resource bounds.
6. Keep errors classified and actionable; partial failure never becomes success.
7. Add or strengthen behavior tests and observe the expected pre-implementation failure when
   practical. Cover happy, boundary, failure, and integration paths.
8. Run `make lint`, `make typecheck`, `make test`, and relevant harness/build checks.
9. Review command, file, network, serialization, secret, and path-traversal risks.
10. Review algorithmic cost, unbounded collections/concurrency, and event-loop blocking.
11. Justify and lock every dependency; separately pin dataset/model revisions.
12. Inspect `git status`, `git diff --check`, and the entire diff.
13. Run a fresh independent reviewer pass for non-trivial changes and address material findings.
14. Report each check as `PASS`, `FAIL`, `NOT RUN`, `BLOCKED`, or `UNVERIFIED`.

Never mark a generated config, absent test, unmeasured benchmark, unrun GPU step, or failed
command as `PASS`.
