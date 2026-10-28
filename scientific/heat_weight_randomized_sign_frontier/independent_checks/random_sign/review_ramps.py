"""Independent raw-word interval replay for the finite classical phase ramp.

This checker imports no project code and does not read producer success reports.
All numerical certification uses the integer outward intervals below.  The raw
Clifford+T receipts and emitted words are the objects being checked.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
import gzip
import hashlib
import json
from math import factorial, isqrt
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]
TIGHT = BASE.parent / "heat_weight_tight_frontier"
SCALE = 1 << 512


def ceildiv(a, b):
    return -((-a) // b)


class I:
    """Closed interval [lo/SCALE, hi/SCALE], with integer directed rounding."""
    __slots__ = ("lo", "hi")

    def __init__(self, x=0, hi=None):
        if hi is None:
            x = F(x)
            self.lo = (x.numerator * SCALE) // x.denominator
            self.hi = ceildiv(x.numerator * SCALE, x.denominator)
        else:
            self.lo, self.hi = x, hi

    def __add__(self, other):
        other = as_i(other)
        return I(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-as_i(other))

    def __rsub__(self, other):
        return as_i(other) - self

    def __mul__(self, other):
        other = as_i(other)
        products = (self.lo * other.lo, self.lo * other.hi,
                    self.hi * other.lo, self.hi * other.hi)
        return I(min(products) // SCALE, ceildiv(max(products), SCALE))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_i(other)
        if other.lo <= 0 <= other.hi:
            raise ArithmeticError("interval division through zero")
        inv = I(SCALE * SCALE // other.hi,
                ceildiv(SCALE * SCALE, other.lo))
        return self * inv

    def square(self):
        high = max(self.lo * self.lo, self.hi * self.hi)
        low = 0 if self.lo <= 0 <= self.hi else min(self.lo * self.lo, self.hi * self.hi)
        return I(low // SCALE, ceildiv(high, SCALE))

    def sqrt(self):
        assert self.lo >= 0
        lo = isqrt(self.lo * SCALE)
        hi = isqrt(self.hi * SCALE)
        if hi * hi < self.hi * SCALE:
            hi += 1
        return I(lo, hi)

    def upper(self):
        return F(self.hi, SCALE)


def as_i(x):
    return x if isinstance(x, I) else I(x)


def certify_pi():
    # Machin's identity and alternating arctangent bounds, independently
    # computed as exact fractions.  Fixed-point conversion is outward.
    def atan_bounds(n):
        partial = sum((F((-1) ** k, (2 * k + 1) * n ** (2 * k + 1))
                       for k in range(260)), F(0))
        return partial, partial + F(1, 521 * n ** 521)
    l5, h5 = atan_bounds(5)
    l239, h239 = atan_bounds(239)
    lower, upper = 16 * l5 - 4 * h239, 16 * h5 - 4 * l239
    return I(I(lower).lo, I(upper).hi)


PI = certify_pi()
IDENTITY_PI_UPPER = F("3.14159265358979323846264338327950288419716939937511")
assert PI.upper() < IDENTITY_PI_UPPER


def sincos(x):
    """Taylor enclosure, |x| <= 4, with an exact uniform remainder."""
    assert max(abs(x.lo), abs(x.hi)) <= 4 * SCALE
    x2 = x * x
    sine, cosine, st, ct = x, I(1), x, I(1)
    for k in range(1, 181):
        st = -(st * x2) / ((2 * k) * (2 * k + 1))
        ct = -(ct * x2) / ((2 * k - 1) * (2 * k))
        sine, cosine = sine + st, cosine + ct
    # Lagrange remainder after degrees 361 and 360, respectively.  Both
    # are enclosed by this deliberately larger exact remainder.
    remainder = I(F(4 ** 361, factorial(361)))
    padding = I(-remainder.hi, remainder.hi)
    return sine + padding, cosine + padding


def c(re=0, im=0):
    return as_i(re), as_i(im)


def cadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def cneg(a):
    return -a[0], -a[1]


def cconj(a):
    return a[0], -a[1]


IDENTITY = ((c(1), c()), (c(), c(1)))
HALFROOT = I(F(1, 2)).sqrt()
GATES = {
    "h": ((c(HALFROOT), c(HALFROOT)), (c(HALFROOT), c(-HALFROOT))),
    "t": ((c(1), c()), (c(), c(HALFROOT, HALFROOT))),
    "tdg": ((c(1), c()), (c(), c(HALFROOT, -HALFROOT))),
    "s": ((c(1), c()), (c(), c(0, 1))),
    "sdg": ((c(1), c()), (c(), c(0, -1))),
    "x": ((c(), c(1)), (c(1), c())),
    "z": ((c(1), c()), (c(), c(-1))),
    "id": IDENTITY,
}
ADJOINT = {"h": "h", "t": "tdg", "tdg": "t", "s": "sdg",
           "sdg": "s", "x": "x", "z": "z", "id": "id"}


def mmul(a, b):
    return tuple(tuple(cadd(cmul(a[r][0], b[0][s]), cmul(a[r][1], b[1][s]))
                       for s in range(2)) for r in range(2))


def replay(word):
    matrix = IDENTITY
    for gate in word:
        matrix = mmul(GATES[gate], matrix)
    return matrix


def error_bound(matrix, j, phase, inverse=False):
    # Rz(theta) = diag(exp(-i theta/2), exp(+i theta/2)).
    sine, cosine = sincos(PI / (2 ** (j + 1)))
    if inverse:
        sine = -sine
    target = ((c(cosine, -sine), c()), (c(), c(cosine, sine)))
    ps, pc = sincos(I(F.from_float(phase)))
    global_phase = c(pc, ps)
    frobenius_squared = I(0)
    for r in range(2):
        for s in range(2):
            difference = cadd(matrix[r][s], cneg(cmul(global_phase, target[r][s])))
            frobenius_squared += difference[0].square() + difference[1].square()
    # Frobenius norm upper-bounds operator norm without a spectral heuristic.
    return frobenius_squared.sqrt().upper()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf8"))


def raw_receipts():
    result, records, hashes = [], {}, {}
    for j in range(3, 21):
        path = TIGHT / "results" / "next" / "qft" / f"rotation_{j:02d}.json.gz"
        row = read_json(path)
        assert row["j"] == j
        word = row["word"]
        assert set(word) <= GATES.keys()
        adjoint_word = [ADJOINT[g] for g in word[::-1]]
        forward = error_bound(replay(word), j, row["comparison_global_phase"])
        backward = error_bound(replay(adjoint_word), j, -row["comparison_global_phase"], True)
        claimed = F(row["interval_operator_upper"])
        assert forward <= claimed, (j, "forward receipt", float(forward), float(claimed))
        assert backward <= claimed, (j, "adjoint receipt", float(backward), float(claimed))
        t_count = sum(g in ("t", "tdg") for g in word)
        assert row["T_count"] == t_count
        digest = sha(path)
        hashes[str(path.relative_to(BASE.parent)).replace("\\", "/")] = digest
        result.append({"j": j, "sha256": digest, "length": len(word),
                       "T_count": t_count, "claimed_operator_upper": str(claimed),
                       "direct_forward_frobenius_upper": str(forward),
                       "direct_adjoint_frobenius_upper": str(backward),
                       "forward_within_receipt": True, "adjoint_within_receipt": True})
        records[j] = {"row": row, "adjoint_word": adjoint_word,
                      "direct_upper": backward, "claimed_upper": claimed}
    return result, records, hashes


def expected_branch(m, s, z, records):
    """Derive a complete canonical word and costs from the branch integer.

    Each wire is treated separately, so interleavings on distinct wires do
    not affect acceptance.  The coefficient check is exact, not numerical.
    """
    assert m >= 1 and s >= 0 and 0 <= z < 2 ** s
    wires, rotations, errors = [[] for _ in range(m)], [], []
    for q in range(m):
        exponent = F(0)
        for r in range(s):
            if not (z >> r) & 1:
                continue
            j = m + s - q - r - 1
            assert j >= 1
            exponent -= F(1, 2 ** j)  # Rz angle measured in units of pi.
            if j == 1:
                word, err = ["sdg"], F(0)
            elif j == 2:
                word, err = ["tdg"], F(0)
            elif j <= 20:
                word, err = records[j]["adjoint_word"], records[j]["claimed_upper"]
            else:
                # ||I-Rz(-pi/2^j)|| = 2 sin(pi/2^(j+2))
                # <= pi/2^(j+1), proved by sin(x) <= x for x >= 0.
                word, err = [], IDENTITY_PI_UPPER / 2 ** (j + 1)
            wires[q].extend(word)
            rotations.append({"q": q, "r": r, "j": j, "gates": len(word)})
            errors.append(err)
        assert exponent == -F(2 * z * 2 ** q, 2 ** (m + s))
    by_wire_t = [sum(g in ("t", "tdg") for g in w) for w in wires]
    counts = Counter(g for w in wires for g in w)
    return {"wires": wires, "rotations": rotations, "counts": counts,
            "T_count": sum(by_wire_t), "gates": sum(map(len, wires)),
            "T_depth": max(by_wire_t, default=0),
            "depth": max(map(len, wires), default=0),
            "operator_error_upper": sum(errors, F(0))}


def check_ramp(path, records, hashes):
    """Check actual branch words and all published scalar ledgers."""
    data = read_json(path)
    assert data["schema"] == "finite_classical_ramp_v1"
    m, S, s = data["m"], data["S"], data["random_bits"]
    assert isinstance(m, int) and m >= 1
    assert isinstance(S, int) and S >= 1 and S & (S - 1) == 0
    assert S == 2 ** s and data["P"] == 2 ** m
    assert data["exponent"] == "-2*pi*i*z*t/(S*P)"
    branches = data["branches"]
    assert len(branches) == S
    assert sorted(b["z"] for b in branches) == list(range(S))
    results = []
    receipt_uses = Counter()
    for b in branches:
        z = b["z"]
        expected = expected_branch(m, s, z, records)
        wires = [[] for _ in range(m)]
        word = b["word"]
        for op in word:
            assert len(op) == 2
            name, q = op
            assert name in GATES and isinstance(q, int) and 0 <= q < m
            wires[q].append(name)
        assert wires == expected["wires"], (path.name, z, "word mismatch")
        inverse_word = [[ADJOINT[op[0]], op[1]] for op in reversed(word)]
        assert b["inverse_word"] == inverse_word
        assert len(b["terms"]) == len(expected["rotations"])
        error = F(0)
        for term, wanted in zip(b["terms"], expected["rotations"]):
            q, r, j = wanted["q"], wanted["r"], wanted["j"]
            assert (term["clock_bit"], term["random_bit"], term["rotation_index"]) == (q, r, j)
            receipt = term["receipt"]
            assert receipt["j"] == j
            receipt_uses[j] += 1
            if j in (1, 2):
                err = F(0)
                assert receipt["kind"] == ("exact S" if j == 1 else "exact T")
            elif j <= 20:
                rawpath = TIGHT / "results" / "next" / "qft" / f"rotation_{j:02d}.json.gz"
                assert Path(receipt["path"]).resolve() == rawpath.resolve()
                assert receipt["sha256"] == sha(rawpath)
                assert receipt["kind"] == "accepted native Rz"
                err = records[j]["claimed_upper"]
            else:
                assert receipt["kind"] == "identity approximation"
                err = IDENTITY_PI_UPPER / 2 ** (j + 1)
                assert F(receipt["operator_error"]) == err
            assert F(term["operator_error"]) == err
            error += err
        assert error == expected["operator_error_upper"]
        assert F(b["operator_error"]) == error
        # Single-qubit gates on different wires can run concurrently.  Each
        # wire's subsequence is serial, so these are exact emitted depths.
        by_wire_T = [sum(g in ("t", "tdg") for g in w) for w in wires]
        actual = {"T": sum(by_wire_T), "gates": len(word),
                  "depth": max(map(len, wires), default=0),
                  "T_depth": max(by_wire_T, default=0)}
        assert actual == {"T": expected["T_count"], "gates": expected["gates"],
                          "depth": expected["depth"], "T_depth": expected["T_depth"]}
        for key, value in actual.items():
            assert b[key] == value, (path.name, z, key)
        results.append({"z": z, **actual, "operator_error_upper": str(error),
                        "term_count": len(b["terms"]), "word_and_inverse_verified": True,
                        "exact_phase_decomposition_verified": True})
    worst = {key: max(b[key] for b in results) for key in ("T", "gates", "depth", "T_depth")}
    mean = {key: sum(F(b[key]) for b in results) / S for key in worst}
    assert data["worst"] == worst
    assert {key: F(value) for key, value in data["mean"].items()} == mean
    worst_error = max(F(b["operator_error_upper"]) for b in results)
    mean_error = sum(F(b["operator_error_upper"]) for b in results) / S
    assert F(data["worst_operator_error"]) == worst_error
    assert F(data["mean_operator_error"]) == mean_error
    all_ones = next(b for b in results if b["z"] == S - 1)
    assert all(all_ones[key] == worst[key] for key in worst)
    assert F(all_ones["operator_error_upper"]) == worst_error
    digest = sha(path)
    hashes[str(path.relative_to(BASE)).replace("\\", "/")] = digest
    return {"file": str(path.relative_to(BASE)).replace("\\", "/"), "sha256": digest,
            "m": m, "P": 2 ** m, "S": S, "branches_checked": S,
            "worst": worst, "mean": {k: str(v) for k, v in mean.items()},
            "worst_operator_error": str(worst_error), "mean_operator_error": str(mean_error),
            "worst_branch": S - 1, "rotation_uses": dict(sorted(receipt_uses.items())),
            "branches": results, "all_checks_pass": True}


def main():
    raw_rows, records, hashes = raw_receipts()
    native = BASE / "results" / "random_sign" / "native"
    ramp_paths = sorted(native.glob("ramp_m*_S*.json.gz"))
    ramps = [check_ramp(path, records, hashes) for path in ramp_paths]
    for source in (Path(__file__).resolve(), BASE / "src" / "random_sign" / "native.py"):
        if source.exists():
            hashes[str(source.relative_to(BASE)).replace("\\", "/")] = sha(source)
    output = {"reviewer": "independent integer fixed-point matrix replay",
              "precision_bits": 512, "imports_project_arithmetic": False,
              "arithmetic": "integer outward intervals; Machin pi; Taylor sin/cos; direct 2x2 gate replay; Frobenius upper bound",
              "raw_receipts_checked": len(raw_rows), "raw_receipts": raw_rows,
              "source_hashes": hashes, "ramp_status": "verified" if ramps else "pending emitted native ramp files",
              "ramp_modules_checked": len(ramps), "branches_checked": sum(r["S"] for r in ramps),
              "ramps": ramps, "all_checks_pass": bool(ramps),
              "phase_convention": "Rz(-pi/2^j) changes the |1>/<0| phase by -pi/2^j; the sum on clock wire q is -2*pi*z*2^q/(S*P). Thus the tensor word is the desired negative ramp up to a branch-global phase.",
              "exact_S_T_convention": "Sdg=e^(-i*pi/4)Rz(-pi/2); Tdg=e^(-i*pi/8)Rz(-pi/4).",
              "identity_rule": "For j>=21 the empty word incurs at most pi_upper/2^(j+1), since 2 sin(pi/2^(j+2)) <= pi/2^(j+1). The pi_upper constant is independently enclosed by Machin's identity.",
              "composition_rule": "The sum of verified primitive operator errors bounds each branch, with branch-global phase discarded only because z is classical. Worst and mean quantities are separately recomputed over all alternatives."}
    target = BASE / "results" / "random_sign" / "review_ramps.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: output[k] for k in ("raw_receipts_checked", "ramp_status", "ramp_modules_checked", "branches_checked", "all_checks_pass")}))


if __name__ == "__main__":
    main()
