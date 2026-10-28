#!/usr/bin/env python3
"""Exact rational checks for the revision review (Python standard library only).

This checks new scalar bounds, not the user's circuit source or hardware.
The classical query bound assumes the manuscript's proved residual-tail bound
and the same ideal public-preprocessing model as its query-complexity theorem.
"""
from __future__ import annotations
import json
import math
from fractions import Fraction as F
from pathlib import Path


def ceil_fraction(x: F) -> int:
    return -((-x.numerator) // x.denominator)


def atan_interval_inverse(q: int, terms: int = 48) -> tuple[F, F]:
    if q <= 1 or terms < 1:
        raise ValueError("q > 1 and terms >= 1 required")
    s = sum((F((-1)**j, (2*j+1)*q**(2*j+1)) for j in range(terms)), F(0))
    other = s + F((-1)**terms, (2*terms+1)*q**(2*terms+1))
    return min(s, other), max(s, other)


def pi_interval() -> tuple[F, F]:
    a, b = atan_interval_inverse(5)
    c, d = atan_interval_inverse(239)
    return 16*a-4*d, 16*b-4*c


def sqrt_upper(x: F, bits: int = 100) -> F:
    if x < 0:
        raise ValueError("nonnegative x required")
    scale = 1 << bits
    q = math.isqrt((x.numerator * scale * scale)//x.denominator)
    if F(q*q, scale*scale) < x:
        q += 1
    ans = F(q, scale)
    assert ans*ans >= x
    return ans


def log_interval(x: F, terms: int = 48) -> tuple[F, F]:
    """Rational enclosure of ln(x): binary range reduction and atanh series."""
    if x <= 0:
        raise ValueError("positive x required")
    if x < 1:
        lo, hi = log_interval(1/x, terms)
        return -hi, -lo
    k = x.numerator.bit_length()-x.denominator.bit_length()
    r = x / F(2)**k
    while r < 1:
        k -= 1
        r *= 2
    while r >= 2:
        k += 1
        r /= 2
    assert 1 <= r < 2 and k >= 0

    def core(z: F) -> tuple[F, F]:
        val = 2*sum((z**(2*j+1)/F(2*j+1) for j in range(terms)), F(0))
        rem = 2*z**(2*terms+1)/(F(2*terms+1)*(1-z*z))
        return val, val+rem

    lo, hi = core((r-1)/(r+1))
    l2, u2 = core(F(1, 3))
    return k*l2+lo, k*u2+hi


def rounded_upper(x: F, denominator: int) -> F:
    return F(ceil_fraction(x*denominator), denominator)


def classical_upper() -> dict:
    N = 1 << 68
    R = 410562  # one selected candidate, not claimed globally optimal
    eta, gamma, delta = F(3,8), F(1,500), F(1,20)
    pilo, pihi = pi_interval()
    # exp(4) >= finite sum; this gives an upper bound for h(pi).
    e4lo = sum((F(4)**j/math.factorial(j) for j in range(65)), F(0))
    bup = 10/(10+e4lo)
    sinup = sqrt_upper(2+sqrt_upper(F(2)))/2  # sin(3 pi / 8)
    kup = bup*sinup/pilo
    assert kup < F(45524522, 10**9)
    # With tau=1, lambda=.1, eta=3/8, D1=1/2,D2=3/2,D3=11/2.
    # h0-b <= h0; b(1-b) <= 1/4; sin(pi eta)<=1.
    e3up = F(11,2)+3*eta*F(3,2)+3*eta**2*F(1,2)+eta**3*F(10,11)+1/(2*pilo)
    assert e3up < 8
    # Integral-test upper bound for the exact paired cardinal tail.
    # cot(z)<=1/z; sin(z)>=z(1-z^2/6); z<=1e-12.
    log_sine_correction = F(1,10**24)
    sm = F(0)
    for shift in (eta, 1-eta):
        zup = pihi*(R+shift)/N
        assert zup < F(1,10**12)
        assert (zup*zup/6)/(1-zup*zup/6) < log_sine_correction
        _, lupper = log_interval(F(N)/(pilo*(R+shift)))
        sm += lupper + 1/(R+shift) + log_sine_correction
    mupper = kup*sm + F(8,(R-1)**2)
    M = rounded_upper(mupper, 10**6)
    _, ln_delta = log_interval(1/delta)
    sample_count = ceil_fraction(2*M*M*ln_delta/(gamma*gamma))
    assert F(sample_count) >= 2*M*M*ln_delta/(gamma*gamma)
    return {
        "scope": "Certified scalar bound conditional on manuscript residual-tail lemma; ideal public-preprocessing query model, not a runtime benchmark",
        "N": str(N), "paired_head_radius": R, "head_queries": 2*R,
        "tail_mass_rational_upper": str(M), "tail_mass_upper_decimal": float(M),
        "sample_queries": sample_count, "total_query_upper": sample_count+2*R,
        "failure_upper": str(delta), "raw_margin": str(gamma),
        "bound": "one-sided Hoeffding; fixed exact-weight importance sampling after a paired head",
        "not_claimed": ["best classical algorithm", "normal approximation certificate", "physical execution", "full generated-label oracle", "global head-radius optimum"],
    }


def main() -> None:
    pilo, pihi = pi_interval()
    Z = F(89698095963077, 2**44)
    gamma, beta, dec = F(1,500), F(3850407,10**11), F(1,2**48)
    g = (gamma-beta)/Z-dec
    qf8192_lo, qf8192_hi = 8192*g/(2*pihi),8192*g/(2*pilo)
    qf16384_lo, qf16384_hi = 16384*g/(2*pihi),16384*g/(2*pilo)
    assert qf8192_hi < F(3,4) < qf16384_lo
    DT, DP, H, Q, L = 137096417,414917110,4096,16383,472711
    res = {
        "status": "Exact rational inequalities checked; no execution of the user's original source",
        "conditional_classical_upper": classical_upper(),
        "phase_gap": {"g":str(g),"g_decimal":float(g),
                      "qf_P8192_interval":[float(qf8192_lo),float(qf8192_hi)],
                      "qf_P16384_interval":[float(qf16384_lo),float(qf16384_hi)],
                      "required_hypothesis":"P*g/(2*pi)>3/4, not >1",
                      "P_for_three_quarter_threshold_interval":[float(F(3,2)*pilo/g),float(F(3,2)*pihi/g)]},
        "serial_model_diagnostics_not_hardware_guarantees": {
            "D_T":DT,"nonoracle_depth":DP,"head":H,"coherent_queries":Q,"classical_query_lower":L,
            "T_only_ratio_zero_oracle_cost":float(F(DT,L-H)),
            "T_only_ratio_equal_oracle_prices":float(F(DT,L-H-Q)),
            "oracle_price_ratio_asymptote":float(F(L-H,Q)),
            "equal_public_layer_model_ratio":float(F(DP,L-H)),
            "T_layers_seconds_at_1_10_100_us":[float(F(DT,10**6)),float(F(DT,10**5)),float(F(DT,10**4))]},
    }
    dst=Path(__file__).resolve().parents[1]/'results'/'exact_rational_checks.json'
    dst.parent.mkdir(exist_ok=True,parents=True)
    dst.write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(json.dumps(res,indent=2))

if __name__=='__main__':
    main()
