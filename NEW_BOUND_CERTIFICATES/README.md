# New finite scalar certificates

Scalar certificates accompanying *Query Bounds and Explicit Quantum Circuits for Periodic Weighted Decisions*.

These standard-library Python programs certify finite scalar inequalities for a specified classical importance-sampling upper bound, phase-gap prerequisites, and explicitly stylized serial-price coefficients. They do not execute a label oracle or a quantum circuit, optimize the classical algorithm, certify physical runtime, or formally verify the manuscript's analytic theorems.

The strengthened uniform-margin theorem (B*) is proved analytically in the manuscript. It is not a formal code proof. The retained finite classical lower-bound certificate is conditional on the analytic lower-bound argument. The new classical upper-bound scalars are conditional on the Fourier residual estimate E3/(R-1)^2, the shifted cotangent integral test, and one-sided per-input Hoeffding as derived in the manuscript.

## Reproduce the new scalars

Use Python 3 with its standard library only. To retain an immutable distribution, extract the archive into a new disposable directory before running these commands from its root:

```sh
python -I -B NEW_BOUND_CERTIFICATES/independent_classical_certificate.py
python -I -B NEW_BOUND_CERTIFICATES/scripts/check_finite_bounds.py
```

The first program writes `NEW_BOUND_CERTIFICATES/independent_classical_certificate.json`; the second writes `NEW_BOUND_CERTIFICATES/results/exact_rational_checks.json`. Those output locations are determined relative to the script files. The distributed JSON files record a fresh isolated replay of the same scripts. Assertions use exact rational arithmetic. Decimal displays are not certificates; the explicitly marked IQAE boundary calculation in the independent output is only a floating-point diagnostic.

The independent program uses an arctangent-addition identity for pi and square-root bisection; the second program uses Machin's identity and scaled integer square roots. Both confirm the paired radius 410562, head count 821124, tail mass upper bound 1505223/500000, sample count 13574839, and total 14395963. The bound is uniform per legal input at margin 1/500 and failure probability at most 1/20, with exact public weights and label-independent preprocessing free. The cotangent integral ends at N/2 in the shifted integration variable. This is one specified classical algorithm, not a claim of optimality.

`revision_bounds.csv` at the archive root separately records the retained classical lower bound 472711, the new specified classical upper bound 14395963, and the compiled quantum upper bound 20479. The original `resources.csv` remains exactly 80 historical rows. All 362 files under `scientific/` retain the bytes of the frozen portable release baseline. The research-source-to-portable-release path and hash-binding transformations remain documented in `provenance.json`; this is not a claim that every portable file equals its original research-repository bytes. The archive's `manifest.json` hashes the payload files; `manifest.sha256` also covers `manifest.json`. Each manifest excludes its own bytes, avoiding a circular hash.

Original executable code is licensed under MIT; original certificate data and documentation are licensed under CC BY 4.0. See the archive-root licence scope and third-party notices.

## Conditional normalization sensitivity

Run `python -I -B NEW_BOUND_CERTIFICATES/check_normalization_sensitivity.py`.
This standard-library program writes the adjacent `normalization_sensitivity.json`.
It encloses the exact tail mass outside the same 4096-label head in
[3.4930414, 3.4930897] and evaluates the complete finite-grid certificate for
four baseline/ideal parameter combinations, without executing a circuit.
The ideal source uses the actual exact mass, exact source/reflection and public
threshold arithmetic, so beta=0; the upper mass endpoint only lower-bounds the
certified gap. P=8192 fails this certificate's precondition even at the largest
possible ideal gap. P=16384, S=4, K=1 gives statistical failure at most
0.002662915301 and, with the unchanged readout-only synthesis allowance
0.000031917874 and conditional logical reserve 0.01, total at most
0.012694833175. The query count remains 20479. The exact source has not been
compiled; no gate, depth, width or preprocessing cost is supplied for it.
All 128 closed base cells and four branches are recorded for each passing case.
The copied finite-grid kernel is bound to its historical source hash. Rational
fields are authoritative; floating-point fields are displays only.
