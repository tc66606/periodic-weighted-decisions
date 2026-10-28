"""Interval-enclosed common head plus cardinal-only tail, rounded to dyadics.

The represented observable has explicit signed head weights and constant
per-side dyadic-shell mass. Its tail acceptance is (L/t)*pi*z*cot(pi*z).
Only the tail's anchored residual is discarded. No label-dependent work occurs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import mpmath as mp


def lo(x):
    return mp.mpf(x._mpi_[0])


def hi(x):
    return mp.mpf(x._mpi_[1])


def enc(x):
    # Decimal export has an explicit guard larger than its final rounding.
    guard = mp.mpf('1e-80') * (1 + max(abs(lo(x)), abs(hi(x))))
    return [mp.nstr(lo(x) - guard, 85), mp.nstr(hi(x) + guard, 85)]


def upper_text(x):
    return mp.nstr(x + mp.mpf('1e-80') * (1 + abs(x)), 85)


def round_dyadic(x, bits):
    integer = int(mp.nint((lo(x) + hi(x)) * mp.mpf(2) ** (bits - 1)))
    dyad = mp.mpf(integer) / mp.mpf(2) ** bits
    return integer, max(abs(dyad - lo(x)), abs(dyad - hi(x)))


def build(N, epsilon='.01', tau='1', lam='.1', eta='.375', bits=40):
    if N < 2 or N & (N - 1):
        raise ValueError('This emitted shell descriptor requires N=2^n.')
    mp.mp.dps = 100
    iv = mp.iv
    iv.dps = 90
    taui, lami, etai, epsi = map(iv.mpf, (str(tau), str(lam), str(eta), str(epsilon)))
    sineta = iv.sin(iv.pi * etai)
    c = sineta / iv.pi
    hp = 1 / (1 + lami * iv.exp(4 * taui))
    strip_x = iv.pi / (4 * taui)
    strip_a = iv.mpf('.9') * iv.ln(strip_x + iv.sqrt(strip_x * strip_x + 1))
    z = iv.exp(-strip_a)
    A = 1 / etai + 1 / (1 - etai)
    U = 4 * c * (A + 2 * iv.ln(5)) + iv.pi ** 2 * sineta
    V = 8 * c

    def B(m):
        return 4 * c * (A + 2 * (1 + iv.ln(4 * m + 1))) + iv.pi ** 2 * sineta

    def trunc(d):
        return z ** (d + 1) * ((U + V * (d + 1)) / (1 - z) + V * z / (1 - z) ** 2)

    d = 1
    while hi(trunc(d)) > lo(epsi / 8):
        d += 1
    M = 2 * d + 1
    while True:
        alias = sum(((z ** (M - m) + z ** (M + m)) / (1 - z ** M)) * B(m)
                    for m in range(1, d + 1))
        if hi(alias) <= lo(epsi / 8):
            break
        M += 1
    samples = [1 / (1 + lami * iv.exp(2 * taui * (1 - iv.cos(2 * iv.pi * j / M))))
               for j in range(M)]
    coefficients = [sum(samples[j] * iv.cos(2 * iv.pi * j * m / M) for j in range(M)) / M
                    for m in range(d + 1)]
    moment = sum(abs(coefficients[m]) * m * m for m in range(1, d + 1))

    def residual_tail(R):
        if 2 * R >= N:
            return iv.mpf(0)
        sums = sum(1 / (R + offset) ** 3 + 1 / (2 * (R + offset) ** 2)
                   for offset in (etai, 1 - etai))
        return 8 * sineta / (3 * iv.pi) * moment * sums

    R = 1 << (2 * d - 1).bit_length()
    while hi(residual_tail(R)) > lo(epsi / 2):
        R *= 2

    def q(r):
        r %= N
        if r > N // 2:
            r -= N
        sign = -1 if r % 2 == 0 else 1
        return sign * sineta / (N * iv.tan(iv.pi * (r - etai) / N))

    def weight(r):
        return hp * q(r) + sum(coefficients[m] *
                              (q(r - m) + q(r + m) - 2 * (-1) ** m * q(r))
                              for m in range(1, d + 1))

    addresses = list(range(N)) if 2 * R >= N else [r % N for j in range(R) for r in (-j, j + 1)]
    heads = []
    head_round = mp.mpf(0)
    for r in addresses:
        value = weight(r)
        integer, err = round_dyadic(value, bits)
        head_round += err
        heads.append({'address': r, 'signed_numerator': integer,
                      'value_interval': enc(value), 'rounding_error_upper': upper_text(err)})
    shells = [] if 2 * R >= N else list(range(R.bit_length() - 1, N.bit_length() - 2))
    shell_numerator, shell_err = round_dyadic(hp * c, bits)
    q_round = 2 * len(shells) * shell_err
    # Every positive/negative side shell has sum(1/t)*f(t/N)<=1.
    public_round = head_round + q_round
    total = hi(trunc(d)) + hi(alias) + hi(residual_tail(R)) + public_round
    if total >= mp.mpf(str(epsilon)):
        raise ArithmeticError('The explicit all-label raw-bias budget was exceeded.')
    norm_numerator = sum(abs(row['signed_numerator']) for row in heads) + 2 * len(shells) * shell_numerator
    return {
        'scope': 'Interval-enclosed, finite dyadic public observable; quantum coin-circuit errors are separate.',
        'N': N, 'n': N.bit_length() - 1, 'tau': str(tau), 'lambda': str(lam), 'eta': str(eta),
        'requested_raw_bias': str(epsilon), 'coefficient_degree': d, 'coefficient_samples': M,
        'common_head_radius': R, 'dyadic_bits': bits, 'dyadic_denominator': str(1 << bits),
        'head': heads,
        'q_tail': {'shell_exponents': shells, 'per_side_shell_mass_numerator': shell_numerator,
                   'per_side_shell_mass_interval': enc(hp * c),
                   'side_address_map': 'side=0: r=-j mod N; side=1: r=j+1; j=2^k+u',
                   'tail_sign': '(-1)^(r+1) * sign(r_canonical-eta)',
                   'tail_acceptance': '(2^k/t) * [pi*(t/N)*cot(pi*t/N)]'},
        'normalization_numerator': str(norm_numerator),
        'normalization': mp.nstr(mp.mpf(norm_numerator) / mp.mpf(2) ** bits, 85),
        'descriptor_entries': len(heads) + 2 * len(shells),
        'compressed_selector_entries': len(heads) + len(shells) + (1 if shells else 0),
        'error': {'anchored_truncation_interval': enc(trunc(d)),
                  'coefficient_alias_interval': enc(alias),
                  'omitted_residual_tail_interval': enc(residual_tail(R)),
                  'head_dyadic_rounding_upper': upper_text(head_round),
                  'q_shell_dyadic_rounding_upper': upper_text(q_round),
                  'all_label_raw_bias_upper': upper_text(total)},
        'coefficient_intervals': [enc(x) for x in coefficients],
        'coefficient_second_absolute_moment_interval': enc(moment),
    }


if __name__ == '__main__':
    output = Path(sys.argv[1])
    exponent = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    epsilon = sys.argv[3] if len(sys.argv) > 3 else '.01'
    result = build(1 << exponent, epsilon=epsilon)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('N', 'coefficient_degree', 'common_head_radius',
                     'descriptor_entries', 'normalization', 'error')}, indent=2))
