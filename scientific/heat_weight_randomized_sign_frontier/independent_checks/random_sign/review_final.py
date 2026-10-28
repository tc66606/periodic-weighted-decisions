"""Consolidate the independent audits after native/program construction freezes.

Run review_ramps.py, review_native_programs.py, review_math.py and
review_program_ledgers.py first. This final binder does not use an author's
success summary as evidence.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/random_sign'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    names=('review_math.json','review_native_programs.json','review_program_ledgers.json','review_resource_menu.json','review_menu.json','review_certificates.json','review_ramps.json')
    receipts={name:read(OUT/name) for name in names}
    for name,row in receipts.items():
        if name=='review_ramps.json':assert row['all_checks_pass'] and row['ramp_status']=='verified'
        else:assert row['status']=='passed'
    programs=receipts['review_program_ledgers.json']['rows']
    assert len(programs)==80
    expected={(tag,response,route,S) for tag in ('discovery_n12','discovery_n68','confirmation_0','confirmation_1')
              for response in ('linear','nodes') for route in ('randomized','padded') for S in (1,2,4,8,16)}
    assert {(r['tag'],r['response'],r['route'],r['S']) for r in programs}==expected
    for row in programs:assert digest(ROOT/row['path'])==row['sha256']
    assert digest(OUT/'resource_menu.json')==receipts['review_resource_menu.json']['sha256']
    assert receipts['review_resource_menu.json']['entries']==1440
    assert digest(OUT/'bound_menu.json')==receipts['review_menu.json']['sha256']
    registration=read(ROOT/'configs/random_sign/confirmation.json')
    registration_hashes={}
    for name,sha in registration['source_sha256'].items():
        assert digest(ROOT/name)==sha;registration_hashes[name]=sha
    for path,sha in registration['inherited_formula_sha256'].items():assert digest(path)==sha
    hashed=hashlib.sha256(registration['fixed_string'].encode()).digest()
    assert hashed.hex()==registration['sha256']
    for index,point in enumerate(registration['points']):
        assert point['n']==(18,30)[index]
        for j,key in enumerate(('tau','lam','eta')):
            assert point['parameters'][key]==registration['choices'][key][hashed[3*index+j]%3]
    targets=[r for r in programs if r['tag']=='discovery_n68' and r['response']=='nodes' and r['S']==4]
    assert len(targets)==2
    conclusions=[]
    for target in targets:
        baseline=target['baseline_resources']
        assert target['longest_chain']<=16383 and target['longest_chain']<baseline['maximum_sequential_label_calls']
        assert F(target['failure'])<F(1,20)
        assert target['total_queries']<baseline['label_queries']
        for new,old in (('T','T_count'),('gates','nonoracle_gates'),('T_depth','T_depth_upper'),('depth','nonoracle_depth_upper')):
            assert target['worst'][new]<baseline[old]
        conclusions.append(dict(route=target['route'],P=target['P'],S=target['S'],K=target['K'],
             failure=target['failure'],worst_resources=target['worst'],total_queries=target['total_queries'],
             longest_chain=target['longest_chain'],allocated_width=target['allocated_width'],baseline_resources=baseline))
    source_path=OUT/'review_sources.json'
    source_validation=dict(status='separate source audit pending',scope='Fresh descriptor/source correctness is not inferred from readout or resource checks.')
    if source_path.exists():
        source=read(source_path)
        assert source['status']=='passed' and source['checked_source_cases']==4
        for path,sha in source['input_sha256'].items():assert digest(path)==sha
        assert digest(ROOT/'independent_checks/random_sign/review_sources.py')==source['checker_sha256']
        source_rows={r['artifact']:r for r in source['rows']}
        for checked in programs:
            if checked['tag'].startswith('confirmation_'):
                program=read(ROOT/checked['path']);ob=program['observable']
                row=source_rows[checked['tag']+'_'+checked['response']]
                assert ob['raw_sha256']==row['raw_sha256'] and ob['native_sha256']==row['native_sha256']
                assert F(ob['Z'])==F(row['exact_Z']) and F(ob['bias'])==F(row['exact_bias'])
        source_validation=dict(status='passed',checked_source_cases=4,path=str(source_path.relative_to(ROOT)),sha256=digest(source_path),
                               scope='Separate established producer-independent source verifier receipt, bound to every fresh program raw/native hash and exact Z/bias. See that receipt for the underlying generic-construction assumptions.')
    report=dict(status='passed',reviewer='Independent skeptical verifier C; fresh-context integer-interval ramp reviewer',
        target_claim='Same promised-sign contract; longest coherent label chain reduced from32767 to16383 with smaller complete worst-branch T, gates and conservative depths.',
        target_conclusions=conclusions,
        coverage=dict(complete_programs=80,full_native_resource_menu_entries=1440,original_statistical_menu_entries=1080,
                      raw_rotation_receipts=receipts['review_ramps.json']['raw_receipts_checked'],
                      ramp_modules=receipts['review_ramps.json']['ramp_modules_checked'],ramp_branches=receipts['review_ramps.json']['branches_checked'],
                      selected_program_certificate_hashes=sorted({r['certificate_sha256'] for r in programs})),
        formal_review=dict(status='no correction required',
            theorem_path='reports/random_sign/FINITE_RANDOMIZED_SIGN_THEOREM.md',
            theorem_sha256=digest(ROOT/'reports/random_sign/FINITE_RANDOMIZED_SIGN_THEOREM.md'),
            checked=['actual finite sine Fourier kernel and removable half-bin points','joint offset/eigenphase and cosine-complement symmetries',
                     'every promised input and head transcript via the raw bias/gap inclusion','asymmetric forward g and backward sqrt(2g) circular separations',
                     'positive near denominators, finite far term, exact decreasing ray tails','complete closed cells with averaging before supremum',
                     'fresh independent offsets for optional majority and once-only logical reserve',
                     'literal ramp sign, shifted finite decoder, padded measurement equivalence and low-only powers']),
        negative_controls=dict(program_corruptions=receipts['review_program_ledgers.json']['negative_controls_rejected'],
             shared_offset_exact=receipts['review_math.json']['shared_offset_counterexample'],
             target_sine_phase_model=receipts['review_math.json']['target_sine_controls']),
        source_validation=source_validation,
        registration=dict(path='configs/random_sign/confirmation.json',sha256=digest(ROOT/'configs/random_sign/confirmation.json'),
                          fixed_formulas_hashes=registration_hashes,deterministic_parameter_rule_verified=True),
        evidence={name:digest(OUT/name) for name in names},
        checker_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'independent_checks/random_sign').glob('review_*.py'))},
        limitations=['Optimality is only within the declared finite certificates and resource menus.',
                     'The bounds are uniform analytic guarantees for ideal circuits plus explicit native readout error and the accepted conditional logical reserve.',
                     'Small actual-word simulations are convention falsification tests, not large quantum executions or all-input statistical evidence.',
                     'Per-input success averages over the algorithm internal random branch; individual nonzero-offset branches need not succeed uniformly.',
                     'Underlying accepted observable correctness is retained by exact source binding; fresh source revalidation is a separate receipt.'])
    (OUT/'review_final.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',programs=80,resource_menu_entries=1440,ramp_branches=report['coverage']['ramp_branches'],
                         target_chain=16383,source_validation=source_validation)))


if __name__=='__main__':main()
