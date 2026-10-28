#!/usr/bin/env python3
"""Portable rational certificate for H4096 tail-normalization sensitivity.

Run with Python >=3.10; only the standard library is used. No accepted archive
is imported or executed. This verifies new scalar inequalities, not circuits.
All certification comparisons use Fraction. Floats only format the report.
The finite-grid kernel below is a bounded copy of the accepted discrete_bound
implementation; provenance and scope are recorded in the output JSON.
"""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import json
import math

PI_LO = F('3.14159265358979323846264338327950288419716939937510')
PI_HI = F('3.14159265358979323846264338327950288419716939937511')
GAMMA = F(1, 500)
DECODER_ERROR = F(1, 2**48)
P_MENU = tuple(2**m for m in range(1, 19))
S_MENU = (1, 2, 4, 8, 16)
K_MENU = tuple(range(1, 128, 2))
BASE_CELLS = 128
RAY_TERMS = 8
A = F(127, 128)
SQRT_BITS = 80
COS_BITS = 64
RAY_BITS = 96
BOUND_BITS = 80


def ceiling(x):
    return -((-x.numerator) // x.denominator)


def up(x, bits=BOUND_BITS):
    return F(ceiling(x * 2**bits), 2**bits)


def sqrt_lower(x):
    assert x >= 0
    n = math.isqrt((x.numerator << (2*SQRT_BITS)) // x.denominator)
    result = F(n, 2**SQRT_BITS)
    assert result**2 <= x < (result + F(1, 2**SQRT_BITS))**2
    return result


def atan_interval(n, count=48):
    s = sum((F((-1)**k, (2*k+1)*n**(2*k+1)) for k in range(count)), F(0))
    next_term = F((-1)**count, (2*count+1)*n**(2*count+1))
    return min(s, s+next_term), max(s, s+next_term)


def exact_side_checks():
    l5, u5 = atan_interval(5)
    l239, u239 = atan_interval(239)
    assert PI_LO < 16*l5-4*u239 < 16*u5-4*l239 < PI_HI
    assert PI_HI**2 < 12
    assert 1-PI_HI**2/(6*16**2) > A
    assert PI_LO/16 - (PI_HI/16)**3/6 > F(3, 16)
    assert PI_HI/2 < 2
    assert F(9, 256)-F(4, 32**2) > 0
    assert F(3, 4)-F(1, BASE_CELLS) > 1/(2*A)
    return True


@lru_cache(None)
def cosine_upper(L, U):
    """Rational upper on max cos^2(pi*f), f in the closed interval [L,U]."""
    assert 0 <= L <= U <= 1
    h = max(abs(L-F(1, 2)), abs(U-F(1, 2)))
    if h == F(1, 2):
        return F(1)
    # cos^2(pi*f)=sin^2(pi*abs(f-1/2)).  Sine increases on [0,pi/2].
    # The degree-17 alternating sine truncation is an upper polynomial.
    sine_upper = sum((
        ((PI_HI*h)**(2*k+1) if k % 2 == 0 else -(PI_LO*h)**(2*k+1))
        / math.factorial(2*k+1) for k in range(9)), F(0))
    assert sine_upper >= 0
    return min(F(1), up(sine_upper**2, COS_BITS))


@lru_cache(None)
def ray_upper(v):
    """Eight exact decreasing terms plus an integral remainder, rounded up."""
    assert v > 1/(2*A)
    direct = sum(((A*A*(v+j)**2-F(1, 4))**-2 for j in range(RAY_TERMS)), F(0))
    D = v + RAY_TERMS-1
    remainder = 1/(3*D**3*(A*A-1/(4*D*D))**2)
    return up(direct+remainder, RAY_BITS)


@lru_cache(None)
def branch_upper(L, U, nf, nb):
    vf, vb = F(nf)+L, F(nb)-U
    C = cosine_upper(L, U)
    bound = up(C*(ray_upper(vf)+ray_upper(vb))/(2*PI_LO**2))
    return vf, vb, C, bound


def certificate(P, S, Z, bias, detailed=False):
    """Uniform sup_phi E_z[bad sign probability], with averaging before sup.

    K repetitions are an optional separate majority operation.  S denotes
    alternative classical branches of a single execution, never S runs.
    """
    assert P in P_MENU and S in S_MENU
    Z, bias = F(Z), F(bias)
    assert Z > 0 and 0 <= bias < GAMMA
    g = (GAMMA-bias)/Z-DECODER_ERROR
    assert 0 < g < 1
    root = sqrt_lower(2*g)
    qf, qb = P*g/(2*PI_HI), P*root/(2*PI_HI)
    result = dict(P=P, S=S, Z=str(Z), bias=str(bias), gamma=str(GAMMA),
                  decoder_error=str(DECODER_ERROR), g=str(g),
                  sqrt_2g_lower=str(root), q_forward=str(qf), q_backward=str(qb),
                  q_forward_float=float(qf), q_backward_float=float(qb))
    if P < 32 or qf <= F(3, 4):
        result.update(status='universal_probability_cap', statistical_failure='1',
                      statistical_failure_float=1.0)
        return result
    far_exact = 8/(F(P)**3*(F(9, 256)-F(4, P**2))**2)
    far = up(far_exact)
    near = F(0)
    worst = None
    cells = []
    for i in range(BASE_CELLS):
        base_L, base_U = F(i, S*BASE_CELLS), F(i+1, S*BASE_CELLS)
        branch_sum = F(0)
        branches = []
        for z in range(S):
            L, U = base_L+F(z, S), base_U+F(z, S)
            nf, nb = ceiling(qf-U), ceiling(qb+L)
            vf, vb, C, value = branch_upper(L, U, nf, nb)
            branch_sum += value
            if detailed:
                branches.append(dict(z=z, L=str(L), U=str(U),
                    forward_integer=nf, backward_integer=nb,
                    v_forward=str(vf), v_backward=str(vb),
                    cosine_squared_upper=str(C),
                    ray_forward_upper=str(ray_upper(vf)),
                    ray_backward_upper=str(ray_upper(vb)), near_upper=str(value)))
        averaged = up(branch_sum/S)
        if worst is None or averaged > near:
            near, worst = averaged, i
        if detailed:
            cells.append(dict(index=i, L=str(base_L), U=str(base_U),
                              averaged_near_upper=str(averaged), branches=branches))
    total = min(F(1), near+far)
    result.update(status='rational_closed_cell_certificate',
                  far_upper=str(far), max_averaged_near_upper=str(near),
                  worst_base_cell=worst, statistical_failure=str(total),
                  statistical_failure_float=float(total))
    if detailed:
        result['cells'] = cells
    return result


KERNEL_SOURCE_SHA256 = 'b1f89b13b5b22c3f5923e7b9b9df9884d60d0aa006b86f2e2e59e9c33669c449'

def decimal_down(x, places=40):
    d = 10**places
    return F((x*d).numerator // (x*d).denominator, d)


def decimal_up(x, places=40):
    return F(ceiling(x*10**places), 10**places)


def interval(lo, hi):
    assert lo <= hi
    return {'lower_exact': str(decimal_down(lo, 30)),
            'upper_exact': str(decimal_up(hi, 30)),
            'lower_display': float(lo), 'upper_display': float(hi)}


def ln_unit_bounds(x, terms=64):
    assert 1 <= x <= 2
    z = (x-1)/(x+1)
    power, total = z, F(0)
    for j in range(terms):
        total += 2*power/(2*j+1)
        power *= z*z
    return total, total+2*power/((2*terms+1)*(1-z*z))


def ln_bounds(x):
    assert x > 0
    if x < 1:
        lo, hi = ln_bounds(1/x)
        return -hi, -lo
    k = 0
    while x >= 2:
        k += 1
        x /= 2
    lo, hi = ln_unit_bounds(x)
    l2, u2 = ln_unit_bounds(F(2))
    return decimal_down(lo+k*l2), decimal_up(hi+k*u2)


def tail_mass():
    N, R, eta = 2**68, 2048, F(3, 8)
    # exp(4): positive series and a geometric bound for the omitted tail.
    e4_lower, term = F(1), F(1)
    for j in range(1, 81):
        term *= F(4, j)
        e4_lower += term
    next_term = term*F(4, 81)
    e4_upper = e4_lower+next_term/(1-F(4, 82))
    b_lower = decimal_down(10/(10+e4_upper))
    b_upper = decimal_up(10/(10+e4_lower))
    sqrt2_lower = sqrt_lower(F(2))
    sqrt2_upper = sqrt2_lower+F(1, 2**SQRT_BITS)
    # sin(3 pi/8)=sqrt(2+sqrt(2))/2, enclosed at both endpoints.
    sin_lower = sqrt_lower(2+sqrt2_lower)/2
    sin_upper = (sqrt_lower(2+sqrt2_upper)+F(1, 2**SQRT_BITS))/2
    kappa_lower = decimal_down(b_lower*sin_lower/PI_HI)
    kappa_upper = decimal_up(b_upper*sin_upper/PI_LO)
    # Uniform E3 bound from Appendix E. h_pi(1-h_pi)<=1/4,
    # sin(pi eta)<=1 and h0-h_pi<=10/11 give the displayed upper.
    E3_upper = F(11,2)+3*eta*F(3,2)+3*eta**2*F(1,2)+eta**3*F(10,11)+1/(2*PI_LO)
    assert E3_upper < 8
    cardinal_lower, cardinal_upper = F(0), F(0)
    components = []
    for alpha in (eta, 1-eta):
        z_hi = PI_HI*(R+alpha)/N
        t_hi = PI_HI*alpha/N
        assert 0 < z_hi < F(1, 10**15)
        v, u = t_hi*t_hi/2, z_hi*z_hi/6
        lower_log, _ = ln_bounds(F(N)/(PI_HI*(R+alpha)))
        _, upper_log = ln_bounds(F(N)/(PI_LO*(R+alpha)))
        # Lower integral: ln cos(pi alpha/N)-ln sin(z).
        lower = lower_log-v/(1-v)
        # First-term + upper integral: (pi/N)cot(z)-ln sin(z).
        upper = upper_log+1/(R+alpha)+u/(1-u)
        cardinal_lower += lower
        cardinal_upper += upper
        components.append({'alpha':str(alpha), 'sum_interval':interval(lower, upper)})
    remainder = F(8, (R-1)**2)
    raw_lower = kappa_lower*cardinal_lower-remainder
    raw_upper = kappa_upper*cardinal_upper+remainder
    # Short exact outward endpoints; true mass is not set to either endpoint.
    mass_lower = decimal_down(raw_lower, 7)
    mass_upper = decimal_up(raw_upper, 7)
    assert 0 < mass_lower <= raw_lower <= raw_upper <= mass_upper
    assert F('3.4930') < mass_lower < mass_upper < F('3.4931')
    return {'N':str(N), 'R':R, 'head_labels':2*R,
            'E3_upper':str(decimal_up(E3_upper, 30)),
            'residual_mass_upper':str(remainder),
            'h_pi':interval(b_lower,b_upper),
            'kappa':interval(kappa_lower,kappa_upper),
            'cardinal_components':components,
            'tail_mass_before_short_rounding':interval(raw_lower,raw_upper),
            'tail_mass':interval(mass_lower,mass_upper)}, mass_lower, mass_upper


def main():
    exact_side_checks()
    mass, ml, mu = tail_mass()
    # Existing accepted randomized P16384,S4,K1 readout receipt. These
    # constants do not certify an ideal tail-source implementation.
    ec = F('70521690043928623137335571290890289504489947137910924705328164012754367/50000000000000000000000000000000000000000000000000000000000000000000000000000000')
    eq = F('144236396808732146084174968234560555793166016036633/10000000000000000000000000000000000000000000000000000000')
    er = F('15338867294903785938781255739787548196375559891551/10000000000000000000000000000000000000000000000000000000')
    synthesis = 2*(ec+eq+er)
    assert synthesis < F('0.000031917874')
    baseline_Z = F(89698095963077, 2**44)
    baseline_beta = F(3850407, 10**11)
    g_ideal_lower = GAMMA/mu-DECODER_ERROR
    g_ideal_upper = GAMMA/ml-DECODER_ERROR
    rows = []
    for name, Z, beta in [('compiled_baseline',baseline_Z,baseline_beta),
                          ('ideal_exact_tail_and_head',mu,F(0))]:
        for P in (8192, 16384):
            row = certificate(P, 4, Z, beta, detailed=True)
            row['scenario'] = name
            row['Z_role'] = 'exact compiled normalization' if beta else 'rational upper bound on exact tail mass; conservative gap input'
            assert F(row['q_backward']) >= F(row['q_forward'])
            if P == 8192:
                # Even the largest gap consistent with the mass interval
                # fails this precondition; this is not an algorithm lower bound.
                best_g = g_ideal_upper if not beta else F(row['g'])
                q_upper = P*best_g/(2*PI_LO)
                assert q_upper < F(3,4)
                assert F(row['statistical_failure']) == 1
                row['q_forward_upper_check'] = str(q_upper)
                row['total_failure_upper'] = '1'
                row['synthesis_evaluation'] = 'unneeded: theorem already returns universal statistical cap; no new P8192 receipt'
            else:
                p = F(row['statistical_failure'])
                total = p+synthesis+F(1,100)
                assert total < F(1,20)
                row['readout_synthesis_failure_upper'] = str(synthesis)
                row['logical_failure_reserve'] = '1/100'
                row['total_failure_upper'] = str(total)
                row['total_failure_upper_display'] = float(total)
                row['total_failure_upper_12dp'] = str(decimal_up(total,12))
                row['statistical_failure_upper_12dp'] = str(decimal_up(p,12))
                row['total_from_displayed_component_uppers'] = str(decimal_up(p,12)+F('0.000031917874')+F('0.01'))
                row['synthesis_evaluation'] = 'same P16384,S4 clock/ramp/QFT; conditional ideal source and inverse exact'
            rows.append(row)
    result = {'status':'PASS', 'arithmetic':'standard-library exact Fraction comparisons; decimal displays non-authoritative',
        'decision_ids':['S04','S52','A04','A05'],
        'kernel_source':'scientific/heat_weight_randomized_sign_frontier/src/random_sign/discrete_bound.py',
        'kernel_source_sha256':KERNEL_SOURCE_SHA256,
        'readout_receipt_source':'scientific/heat_weight_randomized_sign_frontier/results/random_sign/programs/discovery_n68_nodes_randomized_S4.json',
        'readout_receipt_sha256':'de7401d89f04226d495d15f8dbe38863466183afd77d03b8d6d58b22bcc5bdce',
        'tail':mass, 'ideal_gap':interval(g_ideal_lower,g_ideal_upper),
        'ideal_threshold_P':interval(F(3,2)*PI_LO/g_ideal_upper,F(3,2)*PI_HI/g_ideal_lower),
        'readout_synthesis':{'clock_operator':str(ec),'qft_operator':str(eq),
            'ramp_operator_worst':str(er),'two_operator_sum':str(synthesis)},
        'rows':rows,
        'scope':'Conditional same-readout sensitivity with exact tail source/reflection, exact signed weights, exact head arithmetic and exact public normalized-threshold comparison (beta=0); retains 48-bit normalized cosine error and conditional 0.01 whole-program logical reserve. The compiled dyadic head/Z comparison is not transferred to ideal quantities. No sampler, gate, depth, preprocessing, or hardware cost certificate. No old test suite was rerun.'}
    target = Path(__file__).with_name('normalization_sensitivity.json')
    target.write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')
    print('PASS: M_2048 in [%s, %s]' % (ml,mu))
    for row in rows:
        p_integer = ceiling(F(row['statistical_failure'])*10**12)
        p_text = '%s.%012d' % (p_integer//10**12, p_integer%10**12)
        total = F(row.get('total_from_displayed_component_uppers','1'))
        total_integer = ceiling(total*10**12)
        total_text = '%s.%012d' % (total_integer//10**12,total_integer%10**12)
        print('%s P=%s S=4 qf_approx=%.12f p_upper=%s total_upper=%s' % (
            row['scenario'],row['P'],row['q_forward_float'],p_text,total_text))


if __name__ == '__main__':
    main()
