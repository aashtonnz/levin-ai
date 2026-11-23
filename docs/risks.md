# Risks

Last updated: 2026-11-24 (Pacific/Auckland)
Follows: risk list v1.1 (ai-working-practices)

## R-1: Only works on tidy toy worlds
**Why it matters.** Every result so far uses equal square tissues with 4 options. If part-finding or search breaks on irregular, larger or nested structure, the method has no practical use.
**What we can likely do.** Better part-finding (choosing the number of parts, hierarchical clustering); search that scales with the number of parts.
**Early test.** Scale-up test: larger grid, irregular tissues, more options (T-3).

## R-2: Search cost grows too fast with interacting parts
**Why it matters.** When parts depend on neighbours, search tries combinations of settings; with many interacting parts this can grow exponentially.
**What we can likely do.** Use the discovered influence map to keep each search local; fall back to a trained encoder for a first guess.
**Early test.** Count trial runs against the number of parts in the scale-up test, using the shift rule (T-3).

## R-3: Discovered boundaries sometimes misplaced
**Why it matters.** One run in eight finds the right number of parts with wrong boundaries, and steering then fails almost completely.
**What we can likely do.** Have the system check its parts (does search through them reach the examples?) and redo part-finding if not; train the coupling longer.
**Early test.** Part-quality check (T-2).

## R-4: Not new enough to matter
**Why it matters.** The pieces exist separately (neural cellular automata, modular networks, steering inputs). If the combination does not beat existing methods on a real task, it is a curiosity.
**What we can likely do.** Compare against the strongest relevant baselines on one real task.
**Early test.** Literature check and a real-task pilot, after the scale-up test (T-5).

## Settled
- R-5: Training unreliable (1 in 4 seeds failed): settled by keeping each seed's best version and restarting seeds that fail to learn; true-part steering is now 100% on 8 of 8 seeds. 2026-11-24.
