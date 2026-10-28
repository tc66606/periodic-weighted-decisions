"""Two new SHA256-selected public tuples, locked before descriptor evaluation.

The accepted descriptor, response, coordinate, source, and OLD BHMT control
constructors are imported read-only.  This runner writes only its new config
and results directory.  It does not assemble or select randomized readouts.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
ACCEPTED = ROOT.parent/'heat_weight_sign_frontier'
COORD = ROOT.parent/'heat_weight_coordinate_frontier'
OUT = ROOT/'results/random_sign/confirmation'
REG = ROOT/'configs/random_sign/confirmation.json'
FIXED_STRING = 'bounded-polynomial-observables/finite-random-grid-confirmation/2026-09-30'
CHOICES = {'tau': ['.8125', '.9375', '1.0625'],
           'lam': ['.11', '.13', '.17'],
           'eta': ['.21875', '.34375', '.46875']}
LOCKED = ('src/random_sign/discrete_bound.py', 'src/random_sign/native.py',
          'src/random_sign/assemble.py', 'src/random_sign/blind_padding.py')
FORMULAS = (COORD/'src/coordinate/common.py', COORD/'src/coordinate/fresh.py',
            COORD/'src/coordinate/joint.py', COORD/'src/response/experiment.py',
            COORD/'src/response/autonomous_blind_source.py',
            COORD/'src/statistic/blind_compare.py',
            ROOT.parent/'heat_weight_unified_frontier/src/unified/multilevel.py',
            ROOT.parent/'heat_weight_shared_observable/src/shared_observable/blind_practical_descriptor.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, default=str, indent=None if path.suffix == '.gz' else 2)+'\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


def parameters():
    sha = hashlib.sha256(FIXED_STRING.encode('utf8')).digest()
    return sha, [dict(n=n, parameters={key: values[sha[3*j+i] % 3]
                    for i, (key, values) in enumerate(CHOICES.items())})
                 for j, n in enumerate((18, 30))]


def register():
    sha, points = parameters()
    if REG.exists():
        reg = json.loads(REG.read_text())
        assert reg['fixed_string'] == FIXED_STRING and reg['sha256'] == sha.hex()
        assert reg['points'] == points and reg['choices'] == CHOICES
        validate(reg)
        return reg
    reg = dict(schema='finite_random_grid_confirmation_registration_v1',
        fixed_string=FIXED_STRING, sha256=sha.hex(), choices=CHOICES, points=points,
        parameter_rule='SHA256 UTF-8 fixed string; bytes0..2 for n18 and bytes3..5 for n30; byte mod3 indexes ordered tau,lam,eta choices. No unknown label values are read.',
        registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: digest(ROOT/name) for name in LOCKED},
        confirmation_constructor_sha256=digest(__file__),
        inherited_formula_sha256={str(path): digest(path) for path in FORMULAS},
        prior_confirmation_algorithm=dict(path=str(ACCEPTED/'src/sign/confirmation.py'),
                                          sha256=digest(ACCEPTED/'src/sign/confirmation.py')),
        construction_rule='Exactly inherited sign confirmation construction: actual fixed_data descriptor, tail_description, g1..min(8,f), c28,p23, f=min(22,maxshell+eta_bits), round_bits=f-g, merge_small_cut10, nodes shared endpoints and cell_refinement, equal-shell source, parallel coordinate if f even otherwise tapered. Choose minimum complete OLD BHMT T separately for linear and nodes; ties retain lowest g. Preserve every feasible and failed attempt.',
        old_BHMT_menu=dict(P_powers=list(range(1,19)), K=list(range(1,24,2))),
        readout_rule='The locked finite P/S/K construction is evaluated only after these new observables are fixed. This runner does not execute assemble.py or select resource schedules.',
        new_tuple_scope='Genuinely new n18,n30 public tuples; old fresh_0/fresh_1 and heldout_0/heldout_1 are not reused as new evidence.',
        proof_scope='The discrete-window sign theorem is analytic. Full native program validity and resource comparisons remain pending until assembly and independent checks.')
    save(REG, reg)
    print(json.dumps(dict(registration=str(REG), points=points, readout_hashes=reg['source_sha256'])), flush=True)
    return reg


def validate(reg):
    for name, sha in reg['source_sha256'].items():
        assert digest(ROOT/name) == sha, 'Locked readout changed: '+name
    assert digest(__file__) == reg['confirmation_constructor_sha256'], 'Confirmation constructor changed'
    for path, sha in reg['inherited_formula_sha256'].items():
        assert digest(path) == sha, 'Inherited formula changed: '+path


def run(reg):
    validate(reg)
    assert not (OUT/'index.json').exists(), 'Existing complete confirmation is not silently rerun'
    sys.path.insert(0, str(COORD/'src/coordinate'))
    from common import ex, source, thaw, Fraction
    from fresh import make_raw
    from joint import transform
    # These output globals are not used by the pure constructors below, but
    # redirect them defensively so no inherited save helper targets a sibling.
    ex.OUT = OUT
    started = datetime.now(timezone.utc).isoformat()
    save(OUT/'construction_started.json', dict(started_utc=started,
        registration_path=str(REG), registration_sha256=digest(REG),
        descriptor_evaluation_begins_after_registration=True))
    rows = []
    for j, point in enumerate(reg['points']):
        start = time.perf_counter()
        tag = f'confirmation_{j}'
        n = point['n']
        data = ex.ml.fixed_data(1 << n, point['parameters'])
        R = data['radius']
        d, head, pre = ex.api.bc.tail_description(data, R)
        eb = Fraction(d['eta']).denominator.bit_length()-1
        f = min(22, max(d['q_tail']['shell_exponents'])+eb)
        attempts = []
        save(OUT/(tag+'_descriptor.json.gz'), dict(point=point, fixed_data=data,
            tail_descriptor=d, head_coefficients=head, head_preprocessing=pre,
            physical_fraction_bits=f, registration_sha256=digest(REG)))
        for method in ('linear', 'nodes'):
            best = None
            for g in range(1, min(8, f)+1):
                precision = dict(c=28, g=g, p=23, round_bits=f-g, merge_small_cut=10)
                if method == 'nodes':
                    precision['endpoint_layout'] = 'shared'
                b = ex.build(d, method, precision)
                if method == 'nodes':
                    ex.cell_refinement(b, d)
                b = source.transform(b, d)
                bias = data['approximation_bias']+b['bias']
                if bias >= Fraction('.0002'):
                    attempts.append(dict(method=method, g=g, precision=precision,
                        bias=str(bias), feasible=False, reason='raw bias cap',
                        bias_components={k: str(v) for k,v in b.get('bias_components',{}).items()}))
                    save(OUT/(tag+'_attempts.json'), dict(point=point, attempts=attempts, status='running'))
                    print(tag, method, 'g', g, 'failed bias', float(bias), flush=True)
                    continue
                raw = make_raw(b, d, head, pre, data['approximation_bias'])
                raw = transform(raw, 'parallel' if f % 2 == 0 else 'tapered')
                comp = source.compile_source(thaw(raw))
                sch = ex.choose(b['Z'], bias, comp['compiled_ledger']['T_count'])
                if sch is None:
                    attempts.append(dict(method=method, g=g, precision=precision,
                        bias=str(bias), feasible=False, reason='no OLD BHMT menu schedule'))
                    save(OUT/(tag+'_attempts.json'), dict(point=point, attempts=attempts, status='running'))
                    continue
                total = ex.complete_totals(comp, sch['P'], sch['K'], 2*R)
                row = dict(tag=tag, method=method, n=n, parameters=point['parameters'],
                    g=g, precision=precision, bias=str(bias), Z=str(b['Z']), totals=total,
                    schedule=sch, head_preprocessing=pre, physical_fraction_bits=f,
                    descriptor_path=str(OUT/(tag+'_descriptor.json.gz')),
                    descriptor_sha256=digest(OUT/(tag+'_descriptor.json.gz')),
                    selection='Minimum complete OLD BHMT control T over the fixed g menu, separately per response',
                    registration_sha256=digest(REG))
                attempts.append(dict(method=method, g=g, precision=precision, bias=str(bias),
                    feasible=True, T_count=total['T_count'], Z=str(b['Z']),
                    old_BHMT_schedule=sch, complete_totals=total,
                    bias_components={k: str(v) for k,v in b.get('bias_components',{}).items()}))
                save(OUT/(tag+'_attempts.json'), dict(point=point, attempts=attempts, status='running'))
                print(tag, method, 'g', g, 'complete old T', total['T_count'], flush=True)
                if best is None or total['T_count'] < best[0]['totals']['T_count']:
                    best = row, raw, comp
            assert best is not None, 'No feasible source for '+tag+' '+method
            row, raw, comp = best
            name = tag+'_'+method
            row['artifact'] = name
            save(OUT/(name+'_raw.json.gz'), raw)
            save(OUT/(name+'_word.json.gz'), comp)
            row['raw_sha256'] = digest(OUT/(name+'_raw.json.gz'))
            row['native_sha256'] = digest(OUT/(name+'_word.json.gz'))
            save(OUT/(name+'.json'), row)
            rows.append(row)
            print('SAVED', name, row['totals']['T_count'], flush=True)
        save(OUT/(tag+'_attempts.json'), dict(point=point, attempts=attempts, status='complete',
            seconds=time.perf_counter()-start, attempt_count=len(attempts)))
    validate(reg)
    save(OUT/'index.json', dict(rows=rows, registration=reg,
        registration_path=str(REG), registration_sha256=digest(REG),
        started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
        source_scope='Four newly built observable/native source cases; full readout programs are assembled separately.'))
    print('Four new raw/native cases ready:', OUT/'index.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    registration = register()
    if not args.register_only:
        run(registration)
