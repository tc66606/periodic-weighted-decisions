# Periodic weighted decisions: code and certificates

Companion software, data and certificates for **Query Bounds and Explicit Quantum
Circuits for Periodic Weighted Decisions**, by Annan Chen. Software version 2.0.0.

## Version and access identifiers

- Software version: **2.0.0**; version DOI: <https://doi.org/10.5281/zenodo.23736017>.
- Repository: <https://github.com/tc66606/periodic-weighted-decisions>.
- Clone address: `https://github.com/tc66606/periodic-weighted-decisions.git`.
- Manuscript: this release accompanies version 2. Version 1 is on Research Square,
  <https://doi.org/10.21203/rs.3.rs-11513670/v1>.

## Contents

- `scientific/`: 362 source, test, certificate, program and resource files derived
  from the research repositories. Native gate records and parameterized program
  descriptions make the reported logical-resource counts inspectable.
- `resources.csv`: 80 resource rows with complete parameters, relative artifact
  paths and SHA-256 bindings. `controls.csv` retains the comparison schedules.
- `NEW_BOUND_CERTIFICATES/`: standard-library checkers and exact rational outputs
  for the specified classical upper bound and conditional normalization analysis.
- `paper_figures/`: the final two-panel Figure 3 plotting source, dependency pins
  and exact selected values; reads the existing root resource tables.
- `revision_bounds.csv`: the finite classical lower bound, specified classical
  upper bound and compiled quantum upper bound, distinguished by their scope.
- `data_dictionary.json`: field definitions and readable names for retained raw
  labels; `provenance.json`: original/released byte hashes and path transformations.
- `manifest.json` and `manifest.sha256`: integrity records. `CITATION.cff` and
  `release_metadata.json` identify this software version.

The main compiled instance uses n=68, margin 0.002, P=16384, four possible grid
shifts and one execution. It makes 4096 head calls and 20479 total label queries,
with 241883741 public T gates, 137096417 T depth, 414917110 non-oracle depth,
586 allocated logical qubits and a longest chain of 16383 oracle calls.
The failure bound is below 0.039621612312, including a conditional whole-program
logical-failure allowance of 0.01. The opaque oracle's internal implementation,
physical error correction and routing are excluded from the public logical counts.
N=2^68 is an implicit address space, not a stored dataset.

## Minimal reproduction (Python standard library)

Extract into a new directory. From its root:

```sh
python -I -B reproduce.py --mode bindings --output run-bindings
python -I -B NEW_BOUND_CERTIFICATES/independent_classical_certificate.py
python -I -B NEW_BOUND_CERTIFICATES/scripts/check_finite_bounds.py
python -I -B NEW_BOUND_CERTIFICATES/check_normalization_sensitivity.py
```

The binding check verifies all 80 rows against their actual program/certificate
bytes. The three scalar programs write JSON beside their source files, so use a
disposable extracted copy. The first two independently check the specified
14395963-query classical upper bound. The last certifies the exact-tail-mass
enclosure and the finite-grid sensitivity conditional on an ideal exact source.
The ideal source has not been compiled; no new gate/depth/width saving is implied.
These checks do not formally prove the manuscript's analytic theorems.

## Mathematical and complete program/resource replay

The longer, separately scoped modes remain available:

```sh
python -m venv .venv
# Activate the environment using your shell's command.
python -m pip install -r requirements-math.txt
python -I -B reproduce.py --mode math --output run-math
python -m pip install -r requirements-full.txt
python -I -B reproduce.py --mode full --output run-full
```

Use a new output directory for each run. The mathematical mode verifies finite
interval certificates and the retained 472711-query lower-bound certificate,
conditional on the analytic argument in the paper. The full mode additionally
replays source approximations, native gate sequences, adjoints and controls,
finite resource schedules, and small coherent interfaces. It does not execute
or simulate the complete 586-qubit circuit. Each run records step scope, measured
runtime, dependency versions and supported-platform memory measurements.

Worker read guards restrict scientific reads to the extracted copy and installed
Python runtime. They are application-level reproducibility checks, not an OS
sandbox. Dependency versions are pinned, not wheel-hash locked. The recorded
dependency replay used CPython 3.12.14 on Windows; no Linux scientific replay is
claimed. Consult the measured run logs rather than assuming a runtime estimate.

## Scientific scope

The query results concern shared unknown Boolean-label XOR access. The uniform
heat-family theorem fixes the heat/interpolation parameters and covers
N^(-1/2+nu) <= gamma <= gamma0 for fixed 0 < nu < 1/2 and sufficiently large even N.
General multipliers retain a separate fixed-margin result. Analytic proofs are
in the manuscript. The specified classical upper bound is not an optimality
claim or a runtime benchmark. The compiled finite resource table uses its own
normalizations and error certificates; its 80 rows and all 362 files under `scientific/` retain the bytes of the frozen
portable release baseline, including non-winning parameter choices. The earlier
research-source-to-portable-release path and binding transformations are recorded
in `provenance.json`; preservation does not mean all portable files are byte-equal
to their original research-repository counterparts.

## Attribution and rights

Annan Chen, Department of Physics, Imperial College London, London, United Kingdom.
ORCID: https://orcid.org/0009-0000-1308-0466. Contact: annan.chen24@imperial.ac.uk.
The author formulated the original idea and comparison criteria, reviewed and revised
derivations, checked literature and mathematical certificates, ran code, and
reviewed the manuscript. OpenAI Codex and ChatGPT assisted with proof development,
code development, numerical verification and manuscript preparation. The author is
responsible for the content.

Original code is licensed under MIT; original data, certificates and documentation
are licensed under CC BY 4.0. See `LICENSE.md`, `license_map.csv` and
`THIRD_PARTY_NOTICES.md` for the scope and exceptions. Cite the software
using `CITATION.cff`; the Research Square DOI in its references identifies the
earlier manuscript, not this software archive.
