# Test Suite Acceleration Plan

## Goals

Reduce pull-request feedback time and complete-CI wall time without treating tests as duplicates merely because they use the same objective function or problem domain.

The work is organized as four stacked changes:

1. Add runner controls and timing telemetry.
2. Split pull-request CI by responsibility instead of using a Cartesian matrix.
3. Tier and shard optional integration tests, including genuine Open MPI coverage.
4. Reduce workloads in the slowest tests while directly asserting the behavior under test.

## Current Constraints

The standalone test runner executes scripts from both `regression_tests/` and `functionality_tests/`. Those directory names do not reliably describe runtime or dependency requirements, so moving files is not a prerequisite for faster CI.

Many apparently similar tests cover distinct behavior:

- Legacy and standardized generator interfaces.
- Low-level `libE` and `Ensemble` APIs.
- Persistent and nonpersistent generators.
- Different allocation functions and cancellation paths.
- MPI, local, TCP, executor, and resource-management behavior.
- Existing-history evaluation, partial-history restart, and history growth.

The `regression_tests` directory is also referenced by source documentation, literal includes, examples, CI paths, and imports from functionality and unit tests. It will not be emptied as part of this work.

## Runtime Strategy

### Pull Requests

Use Python 3.12 as the canonical integration environment.

Run these jobs independently:

- Static analysis once.
- Unit tests on Python 3.12, 3.13, and 3.14.
- Core local integration tests on Python 3.12.
- Core MPI integration tests on Python 3.12.
- Focused TCP smoke tests.
- Focused macOS smoke tests.
- One canonical coverage job.
- One focused job using a real Open MPI environment.

The exhaustive Python, OS, and transport cross-product remains manually runnable but is not required for pull requests.

### Complete CI

Classify standalone scripts with optional metadata:

- `TESTSUITE_TIER: smoke|core|external|slow`
- `TESTSUITE_FEATURES: transport executor resources optimization redis real-launch`

Shard complete CI by behavior and dependency instead of repeating every test under every transport:

- Core local, MPI, and TCP.
- External integration shards.
- Representative external MPI smoke tests.
- Executors and resources.
- macOS platform behavior.
- Open MPI compatibility.

Start services and install specialized dependencies only in jobs that require them.

### Coverage

Coverage collection is opt-in at the runner level and enabled in one canonical CI job. Other compatibility jobs run without coverage instrumentation. A temporary decrease of no more than 0.5 percentage points is acceptable while the coverage topology changes.

### Failure And Timing Feedback

Pull-request jobs fail fast. Manual diagnostic jobs may continue after failures to report all problems.

The runner records each standalone configuration's filename, communication mode, process count, result, and duration. CI uploads these timings so slow-test changes can be based on median and high-percentile runtime rather than anecdotes.

## Slow-Test Candidates

### gpCAM

Keep legacy and standardized generator coverage, but reduce dimensions, batch size, or batch count after adding direct assertions for covariance sampling, grid reuse, ask/tell ingestion, and persistent feedback.

### Surmise Calibration

Reduce the initial and additional theta populations while retaining initial model construction, at least one emulator rebuild, additional theta generation, and cancellation behavior.

### Variable Resources

Run one substantive variable-resource scheduling case. Exercise additional multiprocessing start methods with smaller smoke workloads and direct assertions.

### MFKG

Keep both batch and asynchronous return modes. Use the minimum budget that completes the initial sample and at least one post-initial acquisition batch.

## Deduplication Policy

Safe consolidation includes:

- Extracting shared setup helpers.
- Removing literally repeated phases with identical inputs and assertions.
- Parameterizing cases where failure isolation and runtime are preserved.
- Retiring legacy coverage only through an explicit deprecation decision.

Tests are not duplicates solely because they share an objective, generator family, domain, or final assertion. Combining files without reducing executed configurations is a maintenance change, not a performance improvement.

## Acceptance Criteria

- Every previously enabled test is mapped to at least one CI job before any deletion.
- Pull-request wall clock targets 3-6 minutes.
- Complete-CI wall clock targets 6-12 minutes.
- Total runner usage decreases by at least 40% for complete CI.
- Changed slow tests directly assert the transitions they are intended to cover.
- Reduced stochastic tests pass repeated local and MPI runs without increased flakiness.
- Documentation references and cross-directory imports remain valid.
