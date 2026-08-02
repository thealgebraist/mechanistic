# Monotonic generation deadline

Transformers 4.57.3 evaluates `MaxTimeCriteria` after each selected token and stops only when `elapsed > max_time`. The C++23 ADT preserves that strict boundary while replacing wall-clock time with `std::chrono::steady_clock` so clock adjustments cannot extend or shorten generation.

All 6 injected boundary cases agree exactly with the pinned Python criterion, including equality, just-over-deadline, zero, and negative-limit cases. All 3 real-audio expired-deadline runs produce the same single token (`1770`), check the deadline once, retain two prefix cache positions, and visit all 74 graph nodes.

This certificate covers finite deadlines on the greedy path. It does not claim deadline scheduling for the separately implemented beam, constrained, contrastive, or prompt-lookup algorithms.
