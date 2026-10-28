"""Replay the four new source certificates with established independent code.

This verifier imports no producer constructor, compiler, saved success verdict,
or resource counter.  The imported response reviewer independently reconstructs
interval coefficients, source rotations, response controls, and native words.
"""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime
import copy
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/'results/random_sign/confirmation'
REPORT = ROOT/'results/random_sign/review_sources.json'
REVIEWER = ROOT.parent/'heat_weight_response_frontier/independent_checks/response/review_final.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_independent():
    spec = importlib.util.spec_from_file_location('accepted_independent_source_review', REVIEWER)
    reviewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reviewer)
    return reviewer


def fixed_data_check(reviewer, witness):
    """Independently enclose both full-head weights and saved residual weights."""
    bc = reviewer.bc
    data = witness['fixed_data']
    base = data['base']
    d = witness['tail_descriptor']
    R = data['radius']
    assert R == base['common_head_radius'] == d['common_head_radius']
    assert d['head'] == []
    for key in ('N', 'n', 'tau', 'lambda', 'eta', 'dyadic_bits', 'dyadic_denominator',
                'coefficient_degree', 'coefficient_samples', 'coefficient_intervals',
                'coefficient_second_absolute_moment_interval'):
        assert base[key] == d[key]
    assert base['q_tail']['per_side_shell_mass_numerator'] == d['q_tail']['per_side_shell_mass_numerator']
    assert base['q_tail']['per_side_shell_mass_interval'] == d['q_tail']['per_side_shell_mass_interval']
    ctx = bc.context(d)
    expected_addresses = {r % d['N'] for j in range(R) for r in (-j, j+1)}
    den = int(d['dyadic_denominator'])
    max_error = {'weights': F(0), 'residuals': F(0)}
    for group in ('weights', 'residuals'):
        rows = data[group]
        assert {int(k) for k in rows} == expected_addresses
        assert {v['address'] for v in rows.values()} == expected_addresses
        for address, row in rows.items():
            address = int(address)
            assert address == row['address']
            value = ctx['residual'](address)
            if group == 'weights':
                value += ctx['hp']*ctx['q'](address)
            lo, hi = bc.bounds(value)
            saved_lo, saved_hi = map(F, row['value_interval'])
            assert saved_lo <= lo <= hi <= saved_hi
            error = bc.error(value, F(row['signed_numerator'], den))
            assert error <= F(row['rounding_error_upper'])
            assert error < F(1, 2*den)
            max_error[group] = max(max_error[group], error)
    expected_head = [data['weights'][str(r % d['N'])] for j in range(R) for r in (-j, j+1)]
    assert witness['head_coefficients'] == expected_head
    global_upper = ctx['global_bound'](R)
    mapping = {'anchored_truncation': 'anchored_truncation_interval',
               'coefficient_alias': 'coefficient_alias_interval',
               'original_residual_tail': 'omitted_residual_tail_interval'}
    components = {}
    for key, field in mapping.items():
        declared = F(base['error'][field][1])
        assert global_upper[key] <= declared
        components[key] = dict(independent_upper=str(global_upper[key]), declared_upper=str(declared))
    fixed = sum((F(base['error'][field][1]) for field in mapping.values()), F(0))
    assert fixed == F(data['approximation_bias'])
    pre = witness['head_preprocessing']
    assert pre['head_entries'] == pre['head_label_queries'] == 2*R
    assert pre['original_descriptor_radius'] == R
    assert pre['extra_cardinal_weight_evaluations'] == 0
    assert F(pre['extra_head_rounding_error_upper']) == 0
    eb = F(d['eta']).denominator.bit_length()-1
    assert witness['physical_fraction_bits'] == min(22, max(d['q_tail']['shell_exponents'])+eb)
    return dict(radius=R, full_head_weights_verified=2*R, residual_weights_verified=2*R,
                stored_intervals_and_nearest_dyadic_rounding_verified=True,
                maximum_rounding_errors={k: str(v) for k,v in max_error.items()},
                approximation_components=components, exact_declared_fixed_bias=str(fixed))


def rejected(action):
    try:
        action()
    except AssertionError:
        return True
    raise AssertionError('Deliberately corrupted certificate was accepted')


def main():
    r = import_independent()
    reg_path = ROOT/'configs/random_sign/confirmation.json'
    reg = r.read(reg_path)
    index = r.read(FOLDER/'index.json')
    assert index['registration'] == reg and index['registration_sha256'] == sha(reg_path)
    seed = 'bounded-polynomial-observables/finite-random-grid-confirmation/2026-09-30'
    digest = hashlib.sha256(seed.encode()).digest()
    choices = {'tau': ['.8125', '.9375', '1.0625'], 'lam': ['.11', '.13', '.17'],
               'eta': ['.21875', '.34375', '.46875']}
    points = [dict(n=n, parameters={k:v[digest[3*j+i] % 3]
                                   for i,(k,v) in enumerate(choices.items())})
              for j,n in enumerate((18,30))]
    assert reg['fixed_string'] == seed and reg['sha256'] == digest.hex()
    assert reg['choices'] == choices and reg['points'] == points
    for path, expected in reg['source_sha256'].items():
        assert sha(ROOT/path) == expected
    assert sha(ROOT/'src/random_sign/confirmation.py') == reg['confirmation_constructor_sha256']
    for path, expected in reg['inherited_formula_sha256'].items():
        assert sha(path) == expected
    started = r.read(FOLDER/'construction_started.json')
    assert started['registration_sha256'] == sha(reg_path)
    assert datetime.fromisoformat(reg['registered_utc']) <= datetime.fromisoformat(started['started_utc']) <= datetime.fromisoformat(index['completed_utc'])
    assert {(x['tag'], x['method']) for x in index['rows']} == {
        (f'confirmation_{j}',method) for j in range(2) for method in ('linear','nodes')}
    fixed_reports = {}
    rows = []
    negative_controls = {}
    for claim in index['rows']:
        name = claim['artifact']
        raw_path = FOLDER/(name+'_raw.json.gz')
        word_path = FOLDER/(name+'_word.json.gz')
        raw, word = r.read(raw_path), r.read(word_path)
        assert sha(raw_path) == claim['raw_sha256'] and sha(word_path) == claim['native_sha256']
        witness_path = Path(claim['descriptor_path'])
        assert sha(witness_path) == claim['descriptor_sha256']
        witness = r.read(witness_path)
        assert witness['registration_sha256'] == claim['registration_sha256'] == sha(reg_path)
        assert witness['tail_descriptor'] == raw['descriptor']
        assert witness['head_coefficients'] == raw['head_coefficients']
        assert witness['head_preprocessing'] == raw['head_preprocessing'] == claim['head_preprocessing']
        assert F(witness['fixed_data']['approximation_bias']) == F(raw['fixed_coefficient_and_tail_bias'])
        j = int(claim['tag'].split('_')[-1])
        assert witness['point'] == points[j]
        assert claim['parameters'] == points[j]['parameters'] and claim['n'] == points[j]['n']
        assert raw['descriptor']['n'] == claim['n']
        assert [raw['descriptor'][k] for k in ('tau','lambda','eta')] == [claim['parameters'][k] for k in ('tau','lam','eta')]
        if claim['tag'] not in fixed_reports:
            fixed_reports[claim['tag']] = fixed_data_check(r, witness)
        eps, source = r.source_replay(raw)
        bias, tables = r.tables_and_bias(raw, eps)
        descriptor = r.descriptor_check(raw)
        native = r.native_word_check(raw, word)
        assert bias == F(raw['raw_bias_fraction']) == F(claim['bias']) < F('.0002')
        assert F(claim['Z']) == F(raw['normalization_fraction']) == F(descriptor['normalization'])
        assert len(raw['head_coefficients']) == claim['totals']['measured_head_queries']
        if not negative_controls:
            damaged = copy.deepcopy(raw)
            damaged['source_receipt']['rotations'][0]['interval_projective_error'] = '0'
            negative_controls['zeroed_source_rotation_error_rejected'] = rejected(lambda: r.source_replay(damaged))
            damaged = copy.deepcopy(raw)
            damaged['raw_bias_fraction'] = str(F(raw['raw_bias_fraction'])+F(1,10**12))
            negative_controls['altered_raw_bias_rejected'] = rejected(lambda: r.tables_and_bias(damaged, eps))
            damaged_word = copy.deepcopy(word)
            damaged_word['compute_inverse'] = damaged_word['compute_inverse'][:-1]
            negative_controls['truncated_literal_inverse_rejected'] = rejected(lambda: r.native_word_check(raw, damaged_word))
        rows.append(dict(artifact=name, exact_Z=claim['Z'], exact_bias=str(bias),
            source_operator_error=str(eps), source=source, tables_and_bias=tables,
            descriptor=descriptor, native=native, raw_sha256=sha(raw_path), native_sha256=sha(word_path)))
        print('passed', name, flush=True)
    attempts = []
    for j in range(2):
        report = r.read(FOLDER/f'confirmation_{j}_attempts.json')
        assert report['status'] == 'complete' and report['attempt_count'] == 16
        assert {(x['method'],x['g']) for x in report['attempts']} == {(method,g) for method in ('linear','nodes') for g in range(1,9)}
        for method in ('linear','nodes'):
            feasible = [a for a in report['attempts'] if a['method'] == method and a['feasible']]
            picked = min(feasible, key=lambda a:a['T_count'])
            claim = next(x for x in index['rows'] if x['tag'] == f'confirmation_{j}' and x['method'] == method)
            assert picked['g'] == claim['g'] and picked['T_count'] == claim['totals']['T_count']
        attempts.append(dict(tag=f'confirmation_{j}',attempts=16,
            failed_bias_attempts=sum(not x['feasible'] for x in report['attempts'])))
    result = dict(status='passed', checked_source_cases=len(rows), rows=rows,
        fixed_data_witnesses=fixed_reports, negative_controls=negative_controls,
        registration_source_hashes_verified=True, fixed_parameter_rule_verified=True,
        registered_before_descriptor_evaluation=True, finite_attempt_records=attempts,
        index_sha256=sha(FOLDER/'index.json'), registration_sha256=sha(reg_path),
        input_sha256=r.INPUTS, prior_independent_input_sha256=r.bc.INPUTS,
        checker_sha256=sha(__file__), established_reviewer_sha256=sha(REVIEWER),
        established_bias_checker_sha256=sha(Path(r.bc.__file__)),
        scope='Only the four genuinely new source cases. Established producer-independent 120-digit outward interval and exact rational code replays coefficient/head descriptors, full fixed_data head and residual witnesses, every response table control and all bias terms, source rotation receipts, literal source word/inverse, native compute/true-adjoint and absolute-control source reflection. No producer constructor/compiler/check routine is imported and no historical experiment or old scan is rerun.',
        limits=['Reusing independent checker code is stronger than trusting producer reports but is not external human peer review.',
            'Exact numerical inequalities are rational comparisons of outward interval endpoints; sparse source simulation is an additional floating-point regression check.',
            'Native reconstruction establishes that the stored native word implements its raw Boolean circuit; the inherited generic coordinate construction theorem is retained rather than exhaustively enumerated at these sizes.',
            'Finite old-control attempt records and the selected minimum are bound and checked for consistency; unselected native words are not all regenerated.',
            'Randomized/padded readout words, full readout budgets, all80 complete programs, and conditional logical reserve are outside this source-only report.'])
    REPORT.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('source replay passed:', REPORT, flush=True)


if __name__ == '__main__':
    main()
