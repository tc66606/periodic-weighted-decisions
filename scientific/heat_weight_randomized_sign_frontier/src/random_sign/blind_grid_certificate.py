"""Independent finite-grid sine certificate and convention falsification tests.

This file reads no prior discovery, and writes nothing.  All certification
arithmetic is rational.  Floating-point checks below only test conventions.
"""
from fractions import Fraction as F
import json
import math

P, M = 16384, 16
Z = F(89698095963077, 17592186044416)
BIAS_UPPER = F(".00003850407")
GAMMA = F(".002")
PI_LO = F("3.14159265358979323846264338327950288419716939937510")
PI_HI = F("3.14159265358979323846264338327950288419716939937511")


def certify_pi():
    def bounds(r):
        lo = sum((-1)**k * r**(2*k+1) / (2*k+1) for k in range(48))
        return lo, lo + r**97/97
    l5, u5 = bounds(F(1, 5))
    l239, u239 = bounds(F(1, 239))
    assert PI_LO < 16*l5-4*u239 < 16*u5-4*l239 < PI_HI


def ray(v, offsets=M):
    a = F(127, 128)
    return (a*a*v*v-F(1, 4))**-2/offsets + 1/(3*v**3*(a*a-F(1, 4)/(v*v))**2)


def certificate():
    certify_pi()
    g = (GAMMA-BIAS_UPPER)/Z-F(1, 2**48)
    assert P*g/(2*PI_HI) > 1
    # Avoid square-root rounding when checking P sqrt(2g)/(2pi)>72.
    assert P*P*2*g > (2*PI_HI*72)**2
    stat = (ray(F(1))+ray(F(72)))/(2*PI_LO**2)
    stat += 8/(F(P)**3*(F(9, 256)-F(4, P*P))**2)
    # A deliberately rounded reservation, not a fabricated native receipt.
    readout_operator_reservation = F(1, 10000)
    total = stat + 2*readout_operator_reservation + F(1, 100)
    assert total < F(1, 20)
    return {
        "P": P, "offsets": M, "gap_lower": str(g),
        "forward_bins_lower": 1, "backward_bins_lower": 72,
        "statistical_failure": str(stat), "statistical_failure_float": float(stat),
        "reserved_readout_operator_error": str(readout_operator_reservation),
        "reserved_total_failure": str(total), "reserved_total_failure_float": float(total),
        "maximum_chain": P-1, "label_queries": 4096+P-1,
        "native_receipt_status": "Offset words still require synthesis receipts; reservation is not a receipt.",
    }


def kernel(size, distances):
    import numpy as np
    d = (distances+size/2) % size-size/2
    z = np.pi*d/size
    with np.errstate(divide="ignore", invalid="ignore"):
        ans = (2/size**2 * np.cos(np.pi*d)**2 * np.cos(z)**2 * np.sin(np.pi/(2*size))**2
               / (np.sin(z)**2-np.sin(np.pi/(2*size))**2)**2)
    ans[np.isclose(np.abs(d), .5, rtol=0, atol=1e-12)] = .5
    return ans


def semantic_tests():
    import numpy as np
    cases, worst, sign_worst = 0, 0., 0.
    for p in (8, 16, 32):
        t = np.arange(p)
        amp = np.sqrt(2/p)*np.sin(np.pi*(t+.5)/p)
        for offsets in (2, 4, 16):
            for x in (0., .1, .5, p/4+.03125, p/2, p-.2, -.1):
                mixture = np.empty(p*offsets)
                for ell in range(offsets):
                    state = amp*np.exp(2j*np.pi*t*(x-ell/offsets)/p)
                    probs = np.abs(np.fft.fft(state)/np.sqrt(p))**2/offsets
                    mixture[ell::offsets] = probs
                padded = np.zeros(p*offsets, complex)
                padded[:p] = amp*np.exp(2j*np.pi*t*x/p)
                coherent = np.abs(np.fft.fft(padded)/np.sqrt(p*offsets))**2
                formula = kernel(p, x-np.arange(p*offsets)/offsets)/offsets
                error = max(float(np.max(np.abs(mixture-coherent))),
                            float(np.max(np.abs(mixture-formula))))
                worst = max(worst, error)
                assert error < 1e-12 and abs(float(mixture.sum())-1)<1e-12
                # Compare the two measurement constructions at arbitrary shifted thresholds.
                values = np.cos(2*np.pi*np.arange(p*offsets)/(p*offsets))
                for threshold in (-1.01, -.9, -.001, 0., .123, .99, 1.01):
                    mask = values <= threshold
                    sign_worst = max(sign_worst, abs(float(mixture[mask].sum()-coherent[mask].sum())))
                cases += 1
    # An explicit phase-only counterexample to leaving the original grid fixed.
    g = float((GAMMA-BIAS_UPPER)/Z-F(1, 2**48))
    q = P*g/(2*math.pi)
    x = P/4-(q-1)-1e-6
    angles = 2*np.pi*np.arange(P)/P
    phi = 2*math.pi*x/P
    bad = np.cos(angles) <= math.cos(phi)-g
    fixed_error = float(kernel(P, x-np.arange(P))[bad].sum())
    fine = np.arange(P*M)/M
    fine_bad = np.cos(2*np.pi*fine/P) <= math.cos(phi)-g
    random_error = float((kernel(P, x-fine)/M)[fine_bad].sum())
    assert fixed_error > .08 and random_error < .025
    return {"small_cases": cases, "maximum_probability_error": worst,
            "maximum_shifted_decision_error": sign_worst,
            "phase_model_fixed_grid_counterexample": {"x": x, "failure": fixed_error,
                                                       "random_grid_failure_same_phase": random_error},
            "scope": "Convention falsification only. Uniform guarantee comes from rational certificate."}


if __name__ == "__main__":
    print(json.dumps({"certificate": certificate(), "semantics": semantic_tests()}, indent=2))
