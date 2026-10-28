"""Native zero-padded sine-clock measurement, without high controlled powers.

padding_module(P, S) is the public assembly API.  It writes generated IQFT words
and module metadata only beneath this new worktree's results/random_sign folder.
The accepted source, clock, inverse-QFT compiler, and rotation library are read
only.  The refined S=4 statistical bound belongs to the separate reviewer; this
module establishes its alternative physical measurement implementation.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
import sys
sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ACCEPTED = ROOT.parent / "heat_weight_sign_frontier"
OUT = ROOT / "results/random_sign"


def read(path):
    p = Path(path)
    data = p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix == ".gz" else data)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    raw = (json.dumps(value, indent=2, default=str) + "\n").encode("utf8")
    p.write_bytes(gzip.compress(raw, mtime=0) if p.suffix == ".gz" else raw)
    return p


def accepted_readout():
    spec = importlib.util.spec_from_file_location("_blind_accepted_readout", ACCEPTED / "src/sign/readout_program.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log2_exact(value):
    if not isinstance(value, int) or value < 1 or value & (value-1):
        raise ValueError("P and S must be positive integer powers of two")
    return value.bit_length()-1


def padding_module(P, S):
    """Return native module metadata including paths, counts, errors, and powers.

    `word_path` points to the complete actual inverse-QFT word for P*S, and the
    accepted sine clock word acts only on local wires 0,...,log2(P)-1.  The root
    assembler adds the unchanged source preparation and repeats its controlled
    iterate according to `powers`.  Counts here contain clock and IQFT only.
    """
    m, r = log2_exact(P), log2_exact(S)
    n = m+r
    if not 2 <= m <= n <= 20:
        raise ValueError("Require 2 <= log2(P) <= log2(P*S) <= 20")
    native = accepted_readout()
    clock_path = ACCEPTED / f"results/sign/blind_clock_m{m:02d}.json.gz"
    clock = read(clock_path)
    if clock["P"] != P or any(q >= m for op in clock["word"] for q in op[1:]):
        raise ValueError("Clock does not respect its low-wire support")
    qft_word = native.inverse_qft(n)  # Never call emit_qft: that writes accepted outputs.
    counts = Counter(op[0] for op in qft_word)
    qft_depth = native.depths(qft_word, n)
    rotations = []
    error = F(0)
    for j in range(3, n+1):
        path = native.QFT / f"rotation_{j:02d}.json.gz"
        rot = read(path)
        if not rot["accepted"]:
            raise ValueError("Unaccepted elementary rotation")
        multiplicity = 3*(n-j+1)
        error += multiplicity*F(rot["interval_operator_upper"])
        rotations.append(dict(j=j, multiplicity=multiplicity, path=str(path),
                              sha256=digest(path), operator_error=rot["interval_operator_upper"]))
    qft = dict(schema="blind_padding_native_iqft_v1", m=n, phase_size=P*S,
               word=qft_word, T_count=counts["t"]+counts["tdg"], gates=len(qft_word),
               **qft_depth, operator_error=str(error), rotation_sources=rotations,
               compiler_path=str(ACCEPTED / "src/sign/readout_program.py"),
               compiler_sha256=digest(ACCEPTED / "src/sign/readout_program.py"),
               scope="Literal accepted inverse_qft(n), emitted only in the new worktree. No initial Hadamards added.")
    word_path = write(f"blind_padding_iqft_m{n:02d}.json.gz", qft)
    module = dict(
        schema="blind_padding_module_v1", P=P, S=S, padded_phase_size=P*S,
        m=m, padding_bits=r, phase_width=n,
        initial_high_bits=dict(wires=list(range(m, n)), state="all zero"),
        clock=dict(path=str(clock_path), sha256=digest(clock_path),
                   local_wires=list(range(m)), T_count=clock["T"], gates=clock["nonoracle_gates"],
                   T_depth=clock["T_depth"], depth=clock["nonoracle_depth"],
                   operator_error=clock["operator_error"]),
        inverse_qft=dict(path=str(word_path), sha256=digest(word_path), local_wires=list(range(n)),
                         T_count=qft["T_count"], gates=qft["gates"], T_depth=qft["T_depth"],
                         depth=qft["depth"], operator_error=str(error)),
        word_path=str(word_path), word_sha256=digest(word_path),
        powers=[dict(control_wire=k, repeat=1 << k,
                     operation="accepted_controlled_iterate_with_outer_port_remapped") for k in range(m)],
        coherent_label_queries=P-1, maximum_sequential_label_calls=P-1,
        decoder=dict(outcome="n=S*j+z", phase_size=P*S,
                     callable="signed_mean(n,P*S,bits=48)",
                     decision="sign(c_H+Z*decoded_cosine); the same tie convention as the randomized route"),
        costs=dict(T_count=clock["T"]+qft["T_count"],
                   gates=clock["nonoracle_gates"]+qft["gates"],
                   T_depth_serial_actual=clock["T_depth"]+qft["T_depth"],
                   depth_serial_actual=clock["nonoracle_depth"]+qft["depth"],
                   T_depth_serial_gate_upper=clock["T_depth"]+qft["T_count"],
                   depth_serial_gate_upper=clock["nonoracle_depth"]+qft["gates"]),
        readout_operator_error=str(F(clock["operator_error"])+error),
        high_zero_invariant="Only low m clock wires are prepared and control G powers. High r wires remain exactly zero until the final IQFT. No high controlled G powers are present.",
        statistical_attribution="S16 sufficient certificate was independently discovered here; S4 refined certificate is supplied and attributed to the separate reviewer.")
    write(f"blind_padding_module_P{P}_S{S}.json", module)
    return module


def ideal_sine(P):
    return np.sqrt(2/P)*np.sin(np.pi*(np.arange(P)+.5)/P)


def align(actual, target):
    phase = np.vdot(target, actual)
    return actual*np.exp(-1j*np.angle(phase))


def shifted_probability(P, S, amp, phi):
    result = np.empty(P*S)
    t = np.arange(P)
    for z in range(S):
        inp = amp*np.exp(1j*t*(phi-2*np.pi*z/(P*S)))
        result[z::S] = np.abs(np.fft.fft(inp)/np.sqrt(P))**2/S
    return result


def semantic_checks():
    """Small actual Clifford+T clock and IQFT checks; ideal G eigenbranches.

    Both eigenstates are kept as orthogonal work branches.  Their probabilities
    are combined with arbitrary initial weights, corresponding to tracing out
    the work register after measurement.  The theorem covers its off-diagonal
    density entries because the entire measurement commutes with G.
    """
    native = accepted_readout()
    rows = []
    for m, S in ((5, 4), (9, 4)):
        P, Q = 1 << m, (1 << m)*S
        module = padding_module(P, S)
        clock = read(module["clock"]["path"])
        qft = read(module["inverse_qft"]["path"])
        zero = np.zeros(Q, dtype=complex); zero[0] = 1
        native_source = native.apply(clock["word"], zero)
        assert np.max(np.abs(native_source[P:])) == 0
        ideal_source = np.zeros(Q, dtype=complex); ideal_source[:P] = ideal_sine(P)
        source_error = float(np.linalg.norm(align(native_source, ideal_source)-ideal_source))
        readout_error = float(F(module["readout_operator_error"]))
        max_state, max_tv, max_equivalence, max_decision = 0., 0., 0., 0.
        max_inverse_error = 0.
        phases = [0., np.pi/P, np.pi/2+.001, np.pi-.001, 2*np.pi-.001]
        mixture_cases = 0
        for phi in phases:
            branch_native, branch_ideal, branch_shift = [], [], []
            for theta in (phi, -phi):
                phased = native_source.copy()
                # Exactly the low-bit powers sum_{k<m} bit_k*2^k; no high powers.
                phased[:P] *= np.exp(1j*np.arange(P)*theta)
                actual = native.apply(qft["word"], phased)
                want_in = ideal_source.copy(); want_in[:P] *= np.exp(1j*np.arange(P)*theta)
                wanted = np.fft.fft(want_in)/np.sqrt(Q)
                state_error = float(np.linalg.norm(align(actual, wanted)-wanted))
                assert state_error <= readout_error+1e-11
                inverse_error = float(np.linalg.norm(native.apply(native.inverse(qft["word"]), actual)-phased))
                assert inverse_error < 1e-11
                max_inverse_error = max(max_inverse_error, inverse_error)
                max_state = max(max_state, state_error)
                branch_native.append(np.abs(actual)**2)
                branch_ideal.append(np.abs(wanted)**2)
                branch_shift.append(shifted_probability(P, S, ideal_sine(P), theta))
            for weight in (0., .1, .5, .83, 1.):
                actual = weight*branch_native[0]+(1-weight)*branch_native[1]
                wanted = weight*branch_ideal[0]+(1-weight)*branch_ideal[1]
                shifted = weight*branch_shift[0]+(1-weight)*branch_shift[1]
                equivalence = float(np.max(np.abs(wanted-shifted)))
                assert equivalence < 1e-12
                max_equivalence = max(max_equivalence, equivalence)
                tv = float(np.sum(np.abs(actual-wanted))/2)
                assert tv <= 2*readout_error+1e-11
                max_tv = max(max_tv, tv)
                values = np.cos(2*np.pi*np.arange(Q)/Q)
                # Includes exact outcome thresholds: both tie conventions and the
                # conservative all-ties-fail event use the same fine outcome.
                for threshold in (-1.01, -.25, 0., .3, 1.01, values[Q//4], values[3]):
                    for mask in (values <= threshold, values < threshold, values == threshold):
                        delta = abs(float(actual[mask].sum()-wanted[mask].sum()))
                        assert delta <= 2*readout_error+1e-11
                        max_decision = max(max_decision, delta)
                        assert abs(float(wanted[mask].sum()-shifted[mask].sum())) < 1e-12
                mixture_cases += 1
        rows.append(dict(P=P, S=S, phase_width=m+log2_exact(S),
                         native_clock_source_error=source_error,
                         high_zero_population_before_iqft=0,
                         phases=len(phases), eigenbranches_per_phase=2,
                         mixture_cases=mixture_cases, maximum_native_state_error=max_state,
                         maximum_native_total_variation=max_tv,
                         maximum_exact_povm_probability_discrepancy=max_equivalence,
                         maximum_shifted_decision_error=max_decision,
                         maximum_native_inverse_error=max_inverse_error,
                         certified_readout_operator_bound=str(module["readout_operator_error"]),
                         clock_sha256=module["clock"]["sha256"], iqft_sha256=module["inverse_qft"]["sha256"]))
    result = dict(status="passed", rows=rows,
                  scope="Actual native Clifford+T low-bit sine clocks and padded IQFTs; ideal relevant G eigenphase actions; small tests falsify conventions. Exact POVM equivalence and synthesis-error proof establish arbitrary inputs.")
    write("blind_padding_semantics.json", result)
    return result


if __name__ == "__main__":
    full = padding_module(16384, 4)
    sixteen = padding_module(16384, 16)
    summary = dict(full_module={k: v for k, v in full.items() if k != "powers"},
                   independently_sufficient_sixteen_module={k: v for k, v in sixteen.items() if k != "powers"},
                   semantic_checks=semantic_checks())
    write("blind_padding_summary.json", summary)
    print(json.dumps(dict(full_module=full["costs"], phase_width=full["phase_width"],
                         readout_operator_error=full["readout_operator_error"],
                         semantics=summary["semantic_checks"]), indent=2))
