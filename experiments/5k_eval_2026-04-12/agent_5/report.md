# Agent 5 Audit Report

## Overall verdict

This pack looks good on the keep side and too harsh on the drop side. I agreed with all 12 kept summaries: most are concise, natural, and actually tell the reader why the paper matters. The main weakness is that the drop policy is discarding several summaries that are already strong but one word over the target, where a rewrite pass would have been better than a hard drop.

## Decision counts

- Kept decisions: 12 agree, 0 disagree
- Dropped decisions: 4 agree, 4 disagree

## Row-by-row audit

1. `2306.10580` - kept - agree. Strong concrete payoff and good compression. Slight title overlap, but the "cuts radar signatures while scanning beams" phrasing earns the keep.
2. `2301.04486` - kept - agree. Clean statement of the main theorem and clearly more useful than the title alone. Slightly short at 9 words, but still strong.
3. `2509.19480` - kept - agree. Natural and concise, with a clear product-style hook. Slight risk of title adjacency because "omni-modal navigation" is already in the title.
4. `2412.20478` - kept - agree. Memorable payoff and a real reason to open the paper. Technical, but still readable as a one-line hook.
5. `q-bio/0505043` - kept - agree. Best summary in the pack. Concrete mechanism, vivid analogy, and much more useful than the title.
6. `2511.09081` - kept - agree. Clear comparative result and good density. "Better than SED-fitting" gives the reader an immediate reason to care.
7. `2211.14612` - kept - agree. Strong theorem-style summary with the real payoff up front. Nicely avoids filler.
8. `2505.14690` - kept - agree. Very short, but it captures the whole hook in plain language. Good example of value beyond a somewhat opaque title.
9. `2312.10722` - kept - agree. Good scientific mechanism and outcome in one sentence. Reads naturally and stays specific.
10. `2409.09070` - kept - agree. Technical but still focused on the main result. "Govern cusps" is compact and informative.
11. `2410.06928` - kept - agree. Dense and specialized, but it does state the classification result rather than paraphrasing the title. Readability is limited, though that may be unavoidable for this topic.
12. `2304.12140` - kept - agree. Clear claim, good specificity, and stronger than the title. Slightly short, but still effective.
13. `2412.04312` - dropped - agree. The candidate states a setup fact, but misses the real headline that the paper solves a long-standing extreme-point problem.
14. `2311.11914` - dropped - disagree. The candidate is strong and specific; it only misses by one word. This should have been rewritten, not discarded.
15. `2202.07718` - dropped - disagree. Same pattern as row 14: good payoff, natural phrasing, just one word too long. Easy salvage.
16. `2511.20431` - dropped - disagree. Strong candidate with a concrete benefit and clear hook. Again, this looks like rewrite material, not drop material.
17. `2602.09899` - dropped - disagree. The candidate is technical, but it still gives a concrete theorem-level payoff and feels keepable for a math paper. I do not think "too narrow" is enough reason to drop it.
18. `2310.12601` - dropped - agree. "Clarifies" is too weak, and the summary undersells the actual contributions around holographic RG, scaling, and boundary correlations.
19. `2305.07192` - dropped - agree. The candidate reports a result that the abstract explicitly presents as already known; it misses the paper's real payoff of exact values.
20. `2405.16120` - dropped - agree. The candidate captures traffic robustness but drops the fairness half of the contribution, which is central here.

## Best summaries

1. `q-bio/0505043` - "Spatial cell patterns can stall viral spread like a firebreak." Excellent payoff-first phrasing, vivid analogy, and minimal title overlap.
2. `2211.14612` - "Optimal control exists for chemotaxis-consumption models even without solution uniqueness." Strong theorem summary with the non-obvious payoff made explicit.
3. `2312.10722` - "Inlet pressure and heterogeneity shift miscible flows from uniform fronts to fingering patterns." Specific, glanceable, and clearly about why the phenomenon matters.
4. `2511.09081` - "Neural networks predict stellar ages, metallicity, and dust from J-PAS photometry better than SED-fitting." High information density without sounding bloated.
5. `2412.20478` - "Fermions can outpace or lag black holes in spinfoam bounce models." Distinctive and intriguing while still faithful to the abstract.

## Concerning summaries or decisions

1. `2311.11914` - Dropped only for 16 words even though the content is strong: "Magnetic fields cut off quantum critical fluctuations in a strange metal, revealing local and itinerant modes."
2. `2202.07718` - Another likely false negative from rigid length filtering: "Computation in physics class widens the gap between students who are and aren't familiar with it."
3. `2511.20431` - Good candidate with a clear systems payoff, but discarded instead of tightened: "BRIC adapts physics controllers to noisy motion plans at test time for consistent, long-term human motion."
4. `2602.09899` - The drop reason feels too strict relative to the rest of the pack; this is technical, but still a valid main-result summary for a specialist math paper.
5. `2509.19480` - Kept summary is acceptable, but it sits close to the title and could have delivered more payoff by emphasizing generalization or robustness instead of modality interchangeability alone.

## Recommendations

- Add a salvage step for summaries that are 16 words and otherwise strong. Several false negatives here could be fixed with a trivial trim.
- Split drop logic into `hard fail` versus `rewrite needed`. Overlength by one word should not be treated the same as vagueness or missing the main payoff.
- Keep watching accepted summaries for title adjacency. Rows like `2509.19480` and, to a lesser extent, `2306.10580` are good enough, but they are closer to paraphrase than the best examples.
- For theorem-heavy math papers, allow narrow but concrete payoff statements when they clearly express the main result. The current drop on `2602.09899` looks less justified than some accepted specialist summaries.
