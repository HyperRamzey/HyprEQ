1.The Sabotage Rule:
"Whenever you write a unit test for a non-trivial invariant (like a ramp, a guard band, or a parser check), you must verify the test discriminates by temporarily sabotaging the feature (or running a 5-line python check) to confirm the test fails without it."

2.The No-Prose-Algebra Rule:
"Never derive ring-buffer index inequalities, filter delay offsets, or floating-point convergence bounds in prose. Always run a quick python3 -c one-liner to check the exact numbers before editing code."

3.The Anomaly-First Rule:
"If a benchmark, counter, or test output looks surprising (e.g., a limiter engaging on >50% of frames when only 0.8 dB of peaks exceed the threshold), assume a bug in the measurement/code first; never rationalize surprising numbers into documentation until verified."

4.The Spec-Before-Code Rule:
"Before implementing a DSP feature, citing prior art, or writing a formula, standard name or citation into docs, research the governing source on the web and read it — at least two independent primary sources, preferring the publisher page, the official docs, or the repository itself over blog summaries. Record in docs what the source actually says (asymmetric versus symmetric, scalar versus frequency-dependent, deprecated symbol, exact licence), and state plainly which parts of our implementation are ours rather than attributed to them. Never write an equation from memory: derive it, then evaluate the limits in the prose (lambda to 0, lambda to infinity) against the code and confirm they agree. Never state a licence without opening the project's own licence file. This exists because shipped docs have already carried a misattributed prior-art claim, a wrong licence, an equation that contradicted its own surrounding sentence, and a prior-art sentence describing a published routine as penalising something it does not penalise. (The paper whose citation was checked turned out to be stated correctly: JAES 64(10) 752-761 (2016) does estimate one regularization amount per response, which is what the doc said.)"

5.The Measure-The-Claim Rule:
"Never mark a reported defect fixed, or a standard's behaviour correct, from the report text, a changelog, or memory. Reproduce it against the working tree and put the number in the commit or docs — the exact dB the analyser reports, the rectangle the code computes versus the rectangle the harness assumes, the patch level where a dependency's message actually changed. A green gate over a check that compares a seed file against itself, or a test that asserts `str.replace` rather than the function under test, is not evidence: verify that the check can fail."

