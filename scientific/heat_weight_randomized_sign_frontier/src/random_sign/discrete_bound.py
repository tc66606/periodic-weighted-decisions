"""Finite random-offset sine-clock sign certificate, using exact rationals.

No sampled phase values enter this producer.  All displayed decimals are
informational; the rational fields are authoritative.  The finite construction
is frozen below and shared across every accepted source/method pair.
"""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
ACCEPTED = ROOT.parent / 'heat_weight_sign_frontier'
OUT = ROOT / 'results/random_sign'
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


def majority_bound(p, K):
    """Exact iid-majority bound; valid also for independently bounded runs."""
    p = F(p)
    assert K in K_MENU and 0 <= p <= 1
    return sum((F(math.comb(K, j))*p**j*(1-p)**(K-j)
                for j in range((K+1)//2, K+1)), F(0))


def construction():
    return dict(P_menu=list(P_MENU), S_menu=list(S_MENU), K_menu=list(K_MENU),
                base_cells=BASE_CELLS, explicit_ray_terms=RAY_TERMS, a=str(A),
                near_divisor=16, far_sine_lower='3/16', pi_lower=str(PI_LO),
                pi_upper=str(PI_HI), sqrt_lower_fractional_bits=SQRT_BITS,
                cosine_upper_fractional_bits=COS_BITS, ray_upper_fractional_bits=RAY_BITS,
                final_upper_fractional_bits=BOUND_BITS,
                cosine_rule='degree-17 alternating sine upper; rational pi endpoints',
                branch_rule='ceil(q_forward-U)+L; ceil(q_backward+L)-U',
                quantifier_order='max_base_cell (sum_z closed_cell_bound / S) + far',
                gamma=str(GAMMA), decoder_error=str(DECODER_ERROR))


def accepted_inputs():
    path = ACCEPTED / 'results/sign/final_comparison.json'
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    rows = [r for r in report['rows'] if r['route'] == 'sine_sign']
    assert len(rows) == 12
    assert len({(r['tag'], r['response']) for r in rows}) == len(rows)
    return path, rows


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-only', action='store_true')
    args = parser.parse_args()
    exact_side_checks()
    input_path, inputs = accepted_inputs()
    metadata = dict(construction=construction(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        accepted_input=str(input_path),
        accepted_input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        exact_side_checks_passed=True,
        guarantee='Every phase, both eigenphases, either promised sign, every reachable head transcript; average over one uniform classical offset before supremum.')
    target_rows = []
    for r in inputs:
        if r['tag'] != 'discovery_n68':
            continue
        for S in S_MENU:
            cert = certificate(16384, S, r['Z'], r['bias'], detailed=True)
            value = dict(**metadata, tag=r['tag'], response=r['response'], certificate=cert)
            filename = f"bound_{r['tag']}_{r['response']}_P16384_S{S}.json"
            save(OUT/filename, value)
            target_rows.append(dict(tag=r['tag'], response=r['response'], P=16384, S=S,
                statistical_failure=cert['statistical_failure'],
                statistical_failure_float=cert['statistical_failure_float'], certificate=filename))
            print(f"{r['tag']} {r['response']} P=16384 S={S}: {cert['statistical_failure_float']:.15g}", flush=True)
    save(OUT/'bound_targets.json', dict(**metadata, rows=target_rows))
    if args.target_only:
        return
    rows = []
    for r in inputs:
        for P in P_MENU:
            for S in S_MENU:
                cert = certificate(P, S, r['Z'], r['bias'])
                rows.append(dict(tag=r['tag'], response=r['response'], **cert))
        print(f"menu complete: {r['tag']} {r['response']}", flush=True)
    save(OUT/'bound_menu.json', dict(**metadata, rows=rows,
        row_count=len(rows),
        majority_rule='sum_{j=(K+1)/2}^K binom(K,j) p^j (1-p)^(K-j)',
        native_rule='statistical majority + 2*K*(clock_error + ramp_error + qft_error) + 1/100',
        branch_rule='ramp_error and resources are worst over S alternative branches, without multiplying by S'))


if __name__ == '__main__':
    main()
