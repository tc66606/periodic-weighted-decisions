"""Independent referee calculations. No project module is imported.

Working interval precision is 110 decimal digits; output decimal endpoints
are rounded outward to 100 places using exact integer/Fraction arithmetic.
The finite formulas are reconstructed from the proofs, and the registered
receipts are treated as assertions to check, not as numerical inputs.
Run with Python -B. Numerical diagnostics are not asymptotic proofs.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import itertools
import json
import math
import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
mp.mp.dps = 125
mp.iv.dps = 110
iv = mp.iv
I = iv.mpf


def binary_fraction(raw):
    sign, mantissa, exponent, _ = raw
    if not mantissa:
        return Fraction(0)
    numerator = -mantissa if sign else mantissa
    return Fraction(numerator << exponent) if exponent >= 0 else Fraction(numerator, 1 << -exponent)


def endpoints(value):
    return tuple(binary_fraction(z) for z in value._mpi_)


def ceil_fraction(value):
    return -((-value.numerator) // value.denominator)


def encoded(value):
    low, high = endpoints(value)
    scale = 10**100
    return [str((low * scale).numerator // (low * scale).denominator) + 'e-100',
            str(ceil_fraction(high * scale)) + 'e-100']


def lower(value):
    return endpoints(value)[0]


def upper(value):
    return endpoints(value)[1]


def exact_interval(low, high):
    # Build an interval directly from exact binary endpoint tuples.
    out = I(0)
    out._mpi_ = (low, high)
    return out


def maximum_interval(values):
    low_index = max(range(len(values)), key=lambda j: lower(values[j]))
    high_index = max(range(len(values)), key=lambda j: upper(values[j]))
    return exact_interval(values[low_index]._mpi_[0], values[high_index]._mpi_[1])


def bernoulli_kl(p, q):
    # Difference-of-log form differs from the production ratio-log form.
    return p*(iv.ln(p)-iv.ln(q)) + (1-p)*(iv.ln(1-p)-iv.ln(1-q))


def public_enclosures(n, radius, tau, lam, eta, sharpened):
    t, l, e = I(tau), I(lam), I(eta)
    h_zero = 1/(1+l)
    h_end = 1/(1+l*iv.exp(4*t))
    amplitude = h_end*iv.sin(iv.pi*e)/iv.pi
    first_derivative = t/2
    second_derivative = first_derivative+t*t
    third_derivative = first_derivative+3*t*t+2*t*t*t
    residual_second = second_derivative+2*e*first_derivative+e*e
    residual_third = (third_derivative+3*e*second_derivative+3*e*e*first_derivative
                      +e**3*(h_zero-h_end)
                      +2*t*h_end*(1-h_end)*iv.sin(iv.pi*e)/iv.pi)
    remainder = residual_third/radius**2 if sharpened else 2*residual_second/radius
    head = h_zero + amplitude*(1/(radius+1-e)+1/(radius+1+e)) + remainder
    variance = (amplitude*iv.sqrt(1/(radius-e)+1/(radius+e))+remainder)**2
    # log(N)-log(2)-log(R+1+eta) is evaluated without forming N.
    mass = 2*amplitude*((n-1)*iv.ln(2)-iv.ln(radius+1+e)-1)-remainder
    norm = (amplitude*(1/e+1/(1-e)+2*(n-1)*iv.ln(2)-iv.ln(e)-iv.ln(1-e))
            +h_zero-h_end+residual_second*iv.pi**2/3)
    return {'c_upper': head, 'V_upper': variance, 'M_lower': mass, 'L_upper': norm}


def majority_failure(k):
    q = Fraction(19, 100)
    return sum(Fraction(math.comb(k, j))*q**j*(1-q)**(k-j)
               for j in range(k//2+1, k+1))


def independent_evaluate(row):
    n, radius = row['n'], row['R']
    t, lam, eta = row['tau'], row['lambda'], row['eta']
    gamma, delta, rho = (I(row[k]) for k in ('gamma', 'delta', 'rho'))
    assert 0 < Fraction(eta) < 1 and Fraction(t) > 0 and Fraction(lam) > 0
    assert 0 < Fraction(row['delta']) < Fraction(1, 2)
    assert 0 < Fraction(row['rho']) < Fraction(1, 2)-Fraction(row['delta'])
    assert Fraction(row['gamma']) > 0 and radius > Fraction(eta)
    values = public_enclosures(n, radius, t, lam, eta, row['improved'])
    head, variance, mass, norm = (values[k] for k in ('c_upper', 'V_upper', 'M_lower', 'L_upper'))
    target = gamma*gamma/(2*iv.ln(1/rho))
    legal = (lower(mass) > 0 and upper(variance) <= lower(target)
             and lower(mass-2*head-4*gamma) >= 0 and (1 << n) >= 2*radius+4)
    if not legal:
        lb = I(0)
    elif row['improved']:
        p_high = (1-head/mass)/2+gamma/mass
        p_low = (1-head/mass)/2-gamma/mass
        maximum_kl = maximum_interval([bernoulli_kl(p_high, p_low), bernoulli_kl(p_low, p_high)])
        gap = 1-2*delta-2*rho*(1-delta)
        lb = 2*gap**2/maximum_kl
        values['KL_maximum'] = maximum_kl
    else:
        lb = I(3)*(1-2*delta-2*rho)**2*mass**2/(32*gamma**2)
    k = 1
    while majority_failure(k) > Fraction(row['delta']):
        k += 2
    p = 2
    def amplitude_error(phase_size):
        return 2*norm*(iv.pi/phase_size+(iv.pi/phase_size)**2)
    if row['improved']:
        while upper(amplitude_error(p)) >= lower(gamma):
            p *= 2
    else:
        while Fraction(p) < upper(8*iv.pi*norm/gamma):
            p *= 2
    values['classical_lower'] = lb
    values['raw_error_bound'] = amplitude_error(p)
    qfail = majority_failure(k)
    values['failure_bound'] = I(qfail.numerator)/qfail.denominator
    checks = []
    # Original output enclosures are checked with exact decimal rationals.
    for name, value in values.items():
        if name in row:
            saved_low, saved_high = (Fraction(z) for z in row[name])
            assert saved_low <= lower(value) <= upper(value) <= saved_high, name
            checks.append(name)
    summary = {
        'n': n, 'R': radius, 'tau': t, 'lambda': lam, 'eta': eta,
        'gamma': row['gamma'], 'rho': row['rho'], 'improved': row['improved'],
        'legal': legal, 'classical_integer_lower': ceil_fraction(lower(lb)),
        'P': p, 'K': k, 'quantum_upper': k*(p-1),
        'separation': bool(legal and lower(lb) > k*(p-1)),
        'enclosure_checks': checks, 'intervals': {key: encoded(val) for key, val in values.items()},
    }
    for name in ('legal', 'classical_integer_lower', 'P', 'K', 'quantum_upper', 'separation'):
        assert summary[name] == row[name], (name, summary[name], row[name])
    assert upper(1-8/iv.pi**2) < Fraction(19, 100)
    assert upper(amplitude_error(p)) < Fraction(row['gamma'])
    return summary


def minimum_radius(row):
    target = I(row['gamma'])**2/(2*iv.ln(1/I(row['rho'])))
    radius = row['R']
    at = public_enclosures(100, radius, row['tau'], row['lambda'], row['eta'], row['improved'])['V_upper']
    previous = public_enclosures(100, radius-1, row['tau'], row['lambda'], row['eta'], row['improved'])['V_upper']
    assert upper(at) <= lower(target) < lower(previous)
    return True


def exact_kl_diagnostics():
    """Adversarial grid around both c endpoints and increasing tail masses."""
    mp.mp.dps = 100
    def relative(p, q):
        return p*mp.log(p/q)+(1-p)*mp.log((1-p)/(1-q))
    def divergences(c, mass, g):
        pp, pm = (1+(-c+2*g)/mass)/2, (1+(-c-2*g)/mass)/2
        return (relative(pp, pm), relative(pm, pp))
    total = 0
    max_ratio = mp.mpf(0)
    for cb, g in [(mp.mpf('.9'), mp.mpf('.002')), (mp.mpf('0'), mp.mpf('.1')), (mp.mpf('3.2'), mp.mpf('.2'))]:
        lower_mass = 2*cb+4*g+mp.mpf('.01')
        endpoint_max = max(divergences(cb, lower_mass, g))
        for centre_fraction in range(-20, 21):
            c = cb*centre_fraction/20
            for scale in [mp.mpf(1), mp.mpf('1.001'), mp.mpf(2), mp.mpf(10), mp.mpf(100)]:
                current = max(divergences(c, lower_mass*scale, g))
                assert current <= endpoint_max + mp.mpf('1e-95')
                max_ratio = max(max_ratio, current/endpoint_max)
                total += 1
    return {'status': 'PASS', 'cases': total, 'maximum_ratio': mp.nstr(max_ratio, 20),
            'proof_basis': 'Joint convexity in c; common binary symmetric channel with correlation M_lower/M.'}


def bernoulli_polynomial_diagnostics():
    # One-query Deutsch parity acceptance: P=(1+t1*t2)/2. Its expectation
    # has degree two, which catches an erroneous degree-T acceptance claim.
    rows = []
    for mu in (Fraction(-3, 4), Fraction(-1, 3), Fraction(0), Fraction(2, 5), Fraction(3, 4)):
        expectation = Fraction(0)
        for t1, t2 in itertools.product((-1, 1), repeat=2):
            acceptance = Fraction(1+t1*t2, 2)
            probability = (1+mu*t1)*(1+mu*t2)/4
            expectation += acceptance*probability
        assert expectation == (1+mu*mu)/2
        # Bernstein F'<=T/sqrt(1-mu^2), T=1.
        assert mu*mu*(1-mu*mu) <= 1
        rows.append([str(mu), str(expectation)])
    # A nonvacuous diffuse prior fixture, checked by an exact binomial sum.
    m, g, rho = 400, Fraction(1, 8), Fraction(1, 10)
    plus_probability = Fraction(5, 8)  # score mean 2 gamma
    leakage = sum(Fraction(math.comb(m, k))*plus_probability**k*(1-plus_probability)**(m-k)
                  for k in range(m+1) if Fraction(2*k-m, m) < g)
    assert leakage < rho
    assert Fraction(1, m) <= lower(I(g.numerator)**2/(g.denominator**2*2*iv.ln(10)))
    return {'status': 'PASS', 'one_query_acceptance_degree': 2, 'exact_averages': rows,
            'diffuse_fixture': {'m': m, 'gamma': str(g), 'leakage': float(leakage),
                                'rho': str(rho), 'both_mean_magnitudes': '.25'}}


def regularity_diagnostics():
    """Direct real-Nyquist sums, not a production Fourier implementation."""
    mp.mp.dps = 60
    eta = mp.mpf('.375')
    def q(n, r):
        return (-1)**(r+1)*mp.sin(mp.pi*eta)*mp.cot(mp.pi*(r-eta)/n)/n
    def multiplier(theta):
        return 1+mp.mpf('.4')*mp.cos(theta)+mp.mpf('.2')*mp.cos(2*theta)
    b = multiplier(mp.pi)
    errors = []
    # Finite trigonometric h: exact convolution is an independent alias test.
    for n in (8, 14, 32, 64):
        error = mp.mpf(0)
        for r in range(-n//2+1, n//2+1):
            spectral = (multiplier(0)+multiplier(mp.pi)*mp.cos(mp.pi*(r-eta))) / n
            spectral += sum(2*multiplier(2*mp.pi*k/n)*mp.cos(2*mp.pi*k*(r-eta)/n)/n
                            for k in range(1, n//2))
            convolution = q(n, r)+mp.mpf('.2')*(q(n, r-1)+q(n, r+1))+mp.mpf('.1')*(q(n, r-2)+q(n, r+2))
            anchored = (b*q(n, r)+mp.mpf('.2')*(q(n, r-1)+q(n, r+1)+2*q(n, r))
                        +mp.mpf('.1')*(q(n, r-2)+q(n, r+2)-2*q(n, r)))
            error = max(error, abs(spectral-convolution), abs(spectral-anchored))
        assert error < mp.mpf('1e-55')
        errors.append({'N': n, 'maximum_identity_error': mp.nstr(error, 8)})
    # b=0 smooth case: m=1+cos theta; residual is a second difference.
    smooth_norms = []
    for n in (64, 256, 1024):
        norm = sum(abs(q(n, r)+(q(n, r-1)+q(n, r+1))/2) for r in range(-n//2+1, n//2+1))
        smooth_norms.append({'N': n, 'L1': mp.nstr(norm, 16)})
    # Interior jump despite zero endpoint: direct Dirichlet formula at eta=0.
    jump_norms = []
    for n in (64, 256, 1024):
        norm = mp.mpf(n//2-1)/n
        norm += sum(abs(mp.sin((n//2-1)*mp.pi*r/n)/(n*mp.sin(mp.pi*r/n))) for r in range(1,n))
        jump_norms.append({'N': n, 'L1': mp.nstr(norm, 16)})
    assert all(mp.mpf(jump_norms[j+1]['L1']) > mp.mpf(jump_norms[j]['L1'])+mp.mpf('.3') for j in range(2))
    # Uniform saturation formula is false even with nonempty two-sign promise.
    dominant = []
    for m in (10, 100, 1000):
        gamma = Fraction(1, m)
        residual_mass = Fraction(1, 10)
        assert 1-residual_mass >= gamma
        dominant.append({'coordinates': m+1, 'gamma': str(gamma), 'M_over_gamma': str((1+residual_mass)/gamma),
                         'exact_queries': 1, 'reason': 'All remaining weights sum to 0.1; sign is first label.'})
    return {'status': 'PASS', 'real_nyquist_alias_identity': errors,
            'zero_endpoint_smooth_L1': smooth_norms, 'zero_endpoint_brickwall_L1': jump_norms,
            'dominant_weight_saturation_counterexamples': dominant}


def main():
    sources = ['theory/next/quantum_lower.md', 'theory/next/regularity_classification.md',
               'theory/next/finite_certificate_proof.md', 'src/frontier_next/finite_bounds.py',
               'configs/next/confirmation.json', 'configs/next/register_confirmation.py', 'results/next/finite_query_certificates.json',
               'results/next/finite_confirmation.json']
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources if (ROOT/name).exists()}
    receipts = json.loads((ROOT/'results/next/finite_query_certificates.json').read_text())
    checked = [independent_evaluate(row) for row in receipts['controls']+receipts['discovery']]
    for row in receipts['discovery']:
        minimum_radius(row)
    config = json.loads((ROOT/'configs/next/confirmation.json').read_text())
    digest = hashlib.sha256(config['seed_string'].encode()).hexdigest()
    assert digest == config['sha256']
    digest_bytes = bytes.fromhex(digest)
    selected_tuples = [
        {'tau': ('.6', '1.2', '1.7')[digest_bytes[3*k] % 3],
         'lambda': ('.04', '.2', '.7')[digest_bytes[3*k+1] % 3],
         'eta': ('.1875', '.4375', '.8125')[digest_bytes[3*k+2] % 3]}
        for k in range(3)]
    assert config['parameters'] == selected_tuples
    assert config['finite_gamma'] == ['.003', '.02']
    confirmation = json.loads((ROOT/'results/next/finite_confirmation.json').read_text())
    assert confirmation.get('registration_seed_sha256', confirmation.get('registration_sha256')) == digest
    if 'registration_file_sha256' in confirmation:
        assert confirmation['registration_file_sha256'] == hashes['configs/next/confirmation.json']
    confirmations = []
    for case in confirmation['rows']:
        assert case['parameters'] in config['parameters'] and case['gamma'] in config['finite_gamma']
        rows = [independent_evaluate(row) for row in case['all_attempts']]
        minimum_radius(case['all_attempts'][0])
        assert rows[-1]['separation'] == case['confirmed']
        assert all(not row['separation'] for row in rows[:-1])
        confirmations.append({'parameters': case['parameters'], 'gamma': case['gamma'], 'confirmed': case['confirmed'],
                              'all_attempts': rows})
    claims = {
        'bernstein_quantum_lower': {'verdict': 'ACCEPT', 'reason': 'degree 2T; G=2F-1 contributes the needed factor 1/2.'},
        'off_promise_global_budget': {'verdict': 'ACCEPT', 'reason': 'Finite model and counter-based truncation; XOR inactive padding uses one query.'},
        'wiener_endpoint_classification': {'verdict': 'ACCEPT', 'reason': 'Absolute convergence validates aliasing and uniform residual tails.'},
        'fixed_gamma_tightness': {'verdict': 'ACCEPT', 'reason': 'Constants and N0 depend on fixed gamma, eta and multiplier; no uniform gamma(N) claim.'},
        'effective_N0_threshold': {'verdict': 'ACCEPT', 'reason': 'Displayed exponential threshold algebraically implies M_lower >= 2 c_upper + 4 gamma.'},
        'exact_KL_maximum': {'verdict': 'ACCEPT', 'reason': 'Joint convexity plus reflection and common symmetric-channel contraction.'},
        'old_n300_certificate': {'verdict': 'ACCEPT', 'reason': 'Independent 110-digit interval arithmetic and 100-place outward serialization.'},
        'new_n68_certificate': {'verdict': 'ACCEPT', 'reason': '472711 classical lower versus 458745 phase-only quantum upper.'},
        'SHA_confirmation_finite_results': {'verdict': 'ACCEPT', 'reason': 'All saved attempts and six final certificates reproduced; seed hash matches.'},
        'SHA_parameter_selection': {'verdict': 'ACCEPT', 'reason': 'SHA byte map independently reconstructed; gamma list is fixed in registration, not selected by hash.'},
        'error_contract': {'verdict': 'ACCEPT', 'reason': 'Exact public preparation, strict E<gamma, total sign decision and independent repetition; no hardware or finite-sampler assertion.'},
    }
    report = {'reviewer': 'fresh_theory_review', 'independent_source_imports': False,
              'working_interval_decimal_digits': 110, 'serialized_decimal_places': 100,
              'serialization': 'Exact Fraction conversion of binary interval endpoints, then integer floor/ceil.',
              'source_sha256': hashes, 'claims': claims, 'finite_certificates': checked,
              'confirmation_seed_sha256': digest, 'confirmations': confirmations,
              'KL_diagnostics': exact_kl_diagnostics(), 'polynomial_diagnostics': bernoulli_polynomial_diagnostics(),
              'regularity_and_counterexamples': regularity_diagnostics(),
              'limitations': ['Finite numerical diagnostics do not prove asymptotic theorems.',
                              'No imported estimator, sampler or baseline code.',
                              'No finite circuit, fault-tolerant budget or hardware claim is checked here.',
                              'SHA seed and file consistency do not prove preregistration time; chronology is a local workflow assertion.']}
    target = ROOT/'results/next/fresh_theory_review.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'checked_saved_rows': len(checked)+sum(len(r['all_attempts']) for r in confirmations),
                      'controls': [(r['n'],r['classical_integer_lower'],r['quantum_upper'],r['separation']) for r in checked[:3]],
                      'best': (checked[3]['n'],checked[3]['classical_integer_lower'],checked[3]['quantum_upper']),
                      'confirmation_exponents': [r['all_attempts'][-1]['n'] for r in confirmations],
                      'report': str(target)}))


if __name__ == '__main__':
    main()
