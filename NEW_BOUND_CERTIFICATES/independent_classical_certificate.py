#!/usr/bin/env python3
"""Independent, standard-library rational audit of the bounded v2 scalars.

No manuscript, archive, or supplied certificate is imported or modified.
The numerical checks assume the analytic inequalities in README.md and the manuscript.
All comparisons/assertions use fractions; decimals are display only.
Pi uses atan(1/2)+atan(1/3), unlike the supplied Machin implementation.
Square roots use rational bisection, unlike its scaled integer square root.
"""
from fractions import Fraction as Q
from pathlib import Path
import json
import math


def ceil(x):
    return (x.numerator + x.denominator - 1) // x.denominator


def down(x, places=40):
    d = 10**places
    return Q((x * d).numerator // (x * d).denominator, d)


def up(x, places=40):
    return Q(ceil(x * 10**places), 10**places)


def atan_bounds(x, n=100):
    assert 0 < x < 1 and n % 2 == 0
    total, power = Q(0), x
    for j in range(n):
        total += (-1)**j * power / (2*j + 1)
        power *= x*x
    # Even partial sums lie below atan; the next positive term is an upper.
    return total, total + power/(2*n + 1)


def pi_bounds():
    a, b = atan_bounds(Q(1, 2))
    c, d = atan_bounds(Q(1, 3))
    # tan(atan(1/2)+atan(1/3))=1 and both angles sum into (0, pi/2).
    return down(4*(a+c)), up(4*(b+d))


def sqrt_bounds(x, steps=150):
    assert x >= 0
    lo, hi = Q(0), max(Q(1), x)
    for _ in range(steps):
        mid = (lo+hi)/2
        if mid*mid < x:
            lo = mid
        else:
            hi = mid
    assert lo*lo <= x <= hi*hi
    return down(lo), up(hi)


def ln_unit_bounds(x, terms=64):
    assert 1 <= x <= 2
    z = (x-1)/(x+1)
    power, total = z, Q(0)
    for j in range(terms):
        total += 2*power/(2*j+1)
        power *= z*z
    remainder = 2*power/((2*terms+1)*(1-z*z))
    return total, total+remainder


def ln_bounds(x):
    assert x > 0
    if x < 1:
        a, b = ln_bounds(1/x)
        return -b, -a
    k = 0
    while x >= 2:
        k += 1
        x /= 2
    a, b = ln_unit_bounds(x)
    c, d = ln_unit_bounds(Q(2))
    return down(a+k*c), up(b+k*d)


def interval_output(lo, hi):
    return {"lower_exact": str(down(lo, 30)), "upper_exact": str(up(hi, 30)),
            "lower_display": float(lo), "upper_display": float(hi)}


def main():
    N, R = 2**68, 410562
    eta, gamma, delta = Q(3,8), Q(1,500), Q(1,20)
    pl, pu = pi_bounds()
    # e^4 lower: positive series, with recurrence and 81 terms.
    e4_lower, term = Q(1), Q(1)
    for j in range(1,81):
        term *= Q(4,j)
        e4_lower += term
    b_upper = up(10/(10+e4_lower))
    _, root2_upper = sqrt_bounds(Q(2))
    _, root_nested_upper = sqrt_bounds(2+root2_upper)
    sin_upper = root_nested_upper/2
    kappa_upper = up(b_upper*sin_upper/pl)
    e3_upper = Q(11,2)+3*eta*Q(3,2)+3*eta**2*Q(1,2)+eta**3*Q(10,11)+1/(2*pl)
    assert e3_upper < 8
    # Paired head [-R+1,R] contains 2R points; tail subset |r|>R-1.
    assert N >= 2*(R-1)+4
    contributions, cardinal_upper = [], Q(0)
    for shift in (eta, 1-eta):
        z_upper = pu*(R+shift)/N
        assert 0 < z_upper < Q(1,10**12)
        u = z_upper*z_upper/6
        # cot(z)<=1/z; sin(z)>=z(1-z^2/6); -ln(1-u)<=u/(1-u).
        _, logarithm_upper = ln_bounds(Q(N)/(pl*(R+shift)))
        component = logarithm_upper + 1/(R+shift) + u/(1-u)
        cardinal_upper += component
        contributions.append({"shift": str(shift), "bound": str(up(component,30))})
    tail_upper = kappa_upper*cardinal_upper + Q(8,(R-1)**2)
    published_mass = Q(1505223,500000)
    assert tail_upper < published_mass
    assert up(tail_upper,6) == published_mass
    ll, lu = ln_bounds(1/delta)
    sample_lo = 2*published_mass**2*ll/gamma**2
    sample_hi = 2*published_mass**2*lu/gamma**2
    m = 13574839
    assert m-1 < sample_lo <= sample_hi < m
    assert 2*R + m == 14395963
    Z, beta, dec = Q(89698095963077,2**44), Q(3850407,10**11), Q(1,2**48)
    gap = (gamma-beta)/Z-dec
    q8192 = (8192*gap/(2*pu),8192*gap/(2*pl))
    q16384 = (16384*gap/(2*pu),16384*gap/(2*pl))
    assert q8192[1] < Q(3,4) < q16384[0]
    threshold = (Q(3,2)*pl/gap,Q(3,2)*pu/gap)
    DT, DP, H, coherent, L = 137096417,414917110,4096,16383,472711
    ratios = {"T_zero_coherent_price":Q(DT,L-H),
              "T_equal_oracle_prices":Q(DT,L-H-coherent),
              "coherent_to_classical_price_asymptote":Q(L-H,coherent),
              "public_layers_zero_coherent_price":Q(DP,L-H)}
    # Non-certificate diagnostic only: strict slack should be chosen in a run.
    eps, alpha = float(gap/2), 0.0296
    iqae_diagnostic = 50/eps*math.log(2/alpha*math.log2(math.pi/(4*eps)))
    result = {"status":"PASS", "arithmetic":"all certifying comparisons exact rational",
              "analytic_scope":"conditional on analytic derivations in the manuscript; ideal public-preprocessing label-query model",
              "pi":interval_output(pl,pu), "E3_upper":str(up(e3_upper,30)),
              "kappa_upper":str(kappa_upper), "N":str(N), "R":R,
              "head_queries":2*R,"tail_component_bounds":contributions,
              "tail_upper_before_rounding":str(up(tail_upper,30)),
              "tail_mass_rational_upper":str(published_mass),
              "samples_real_threshold":interval_output(sample_lo,sample_hi),
              "sample_queries":m,"total_query_upper":2*R+m,
              "g":str(gap),"q8192":interval_output(*q8192),
              "q16384":interval_output(*q16384),
              "P_three_quarter_threshold":interval_output(*threshold),
              "runtime_coefficients":{k:{"exact":str(v),"display":float(v)} for k,v in ratios.items()},
              "IQAE_boundary_diagnostic_not_certificate":{"epsilon":eps,"alpha":alpha,"theorem_Q_bound_display":iqae_diagnostic,
                "caution":"excludes initialization/predicate shot costs; epsilon=g/2 is only a boundary diagnostic"}}
    target = Path(__file__).with_name('independent_classical_certificate.json')
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
