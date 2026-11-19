## Lean code style

Adapted from the PufferLib SKILL_ISSUES.md refactoring guide, with a safety exception.

Goal: reduce source length without golfing, while preserving behavior, performance and determinism. Long files or functions are fine when they are simpler to follow. Apply this to new changes; do not refactor unrelated code.

Refactoring pass:
1. Inline every function that is used only once, and tighten the former call site.
2. Flatten deeply nested code by merging and inverting conditionals (early returns).
3. Co-optimize a shared function together with its callers instead of adding parameters and flags.
4. Remove dead code, unused options and needless abstractions before adding anything.
5. Apply the style rules below as a final pass.

Safety exception (overrides the above): never remove or weaken checks that protect money, orders, positions, credentials or user data, such as live exchange reads, leverage and margin confirmation, refusing to act when a read fails, idempotency guards, rate limits and input validation at API boundaries. Do not replace these with asserts that crash mid-operation. Plain asserts are fine for internal invariants where a crash cannot leave an order, position or record half done. Elsewhere, replace convoluted error handling with the simplest handling that still fails safe.

Style:
- Do not split code into more files than needed.
- Soft 80-column, hard 100-column limit.
- Do not one-line loops or conditionals.
- Use semantic vertical spacing between blocks sparingly.
- Do not add source complexity or shims only to make testing easier.
- Tests matter, but their length does not count against the diff budget.
- Do not block off comments with --- or ### lines.
- When the repo's formatter or lint config disagrees, follow the formatter.

Diffs: prefer the smallest diff that fully solves the problem, report the line count, and trim a fix that balloons before review.
