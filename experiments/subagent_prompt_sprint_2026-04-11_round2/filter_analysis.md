# Round 2 notation follow-up

## Prompt-only variants

- `deepseek_v32_generator_round2_notation_policy_v1`
  - `20260407`: `87 kept / 13 dropped`
  - `20260411`: `78 kept / 22 dropped`
  - Read: improved some formula-shaped summaries, but destabilized payoff capture and yielded too many misses on the second seed.

- `deepseek_v32_generator_round2_notation_examples_v1`
  - `20260407`: `84 kept / 16 dropped`
  - `20260411`: `80 kept / 20 dropped`
  - Read: narrower than the full notation-policy prompt, but still worse than the baseline on both seeds.

Conclusion: prompt-side notation steering did not beat the current baseline generator robustly.

## Code-side filter conclusion

Keep the current baseline generator prompt and add a narrow pre-review local filter for obviously formula-heavy notation. The filter targets summary shapes such as:

- LaTeX-style fragments like `$...$` or `\...`
- big-O or similar asymptotic notation like `O(...)`
- ASCII transliterations like `x^(7/3)` or `sigma^-1/2`
- blackboard-style finite-field tokens like `𝔽₉`

This is intentionally narrower than a generic "technical language" filter. It is meant to catch the ugliest feed-hostile summaries without dropping normal technical names.

## Frozen-baseline counterfactual

Applying the new filter to the existing baseline generations would have removed only a small number of accepted rows:

- `20260407`: `4` formula-heavy rows
- `20260411`: `2` formula-heavy rows
- `20260412`: `1` formula-heavy row

Representative filtered summaries:

- `Low-density Fermi mixtures reveal a $x^{7/3} \ln x$ energy term beyond mean-field predictions.`
- `Simplex method now has optimal smoothed complexity with O(σ^{-1/2}) noise dependence.`
- `Generalized quantum channels yield Berezin transforms as trace limits for $SU(2)$.`
- `New additive codes beat best-known linear codes over 𝔽₉ and 𝔽₄.`

Conclusion: the robust win is not a new generator prompt. The robust win is keeping the baseline prompt and letting code drop the worst formula-shaped outputs before review.
