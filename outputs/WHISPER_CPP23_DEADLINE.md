# Monotonic generation deadline

Transformers 4.57.3 stops only when `elapsed > max_time`. Ordinary search loops evaluate `MaxTimeCriteria` after a selected transition, while assisted prompt lookup also evaluates it on the candidate prefix. The C++23 ADT preserves both placements while replacing wall-clock time with `std::chrono::steady_clock` so clock adjustments cannot extend or shorten generation.

All 6 injected boundary cases agree exactly with the pinned Python criterion, including equality, just-over-deadline, zero, and negative-limit cases. All 3 real-audio greedy expired-deadline runs produce the same single token (`1770`), check the deadline once, retain two prefix cache positions, and visit all 74 graph nodes.

All 6 separately converted search interpreters also match their pinned expired-deadline references: standard beam, sampled beam, diverse-group beam, constrained beam, and contrastive search admit token `1770`; prompt lookup admits no target token because the assisted loop checks its candidate prefix first. Every interpreter records one deadline check and visits all 74 graph nodes.

This certificate covers finite deadlines. It remains finite source-pinned evidence, not a real-time scheduling proof for every machine load.
