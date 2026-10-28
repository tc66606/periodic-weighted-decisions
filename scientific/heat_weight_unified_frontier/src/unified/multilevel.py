"""Fixed-coefficient residual telescoping, literal circuits and finite schedules.

The existing control implementation is imported read-only, never copied.
This is a candidate constructor, not an independent checker.
"""
from __future__ import annotations
import argparse
import bisect
from collections import Counter
import copy
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import mpmath as mp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT.parent / 'heat_weight_shared_observable'
QFT_ROOT = ROOT.parent / 'heat_weight_tight_frontier/results/next/qft'
sys.path.insert(0, str(CONTROL / 'src'))
from shared_observable.blind_practical_descriptor import build as control_descriptor, enc, hi, lo, round_dyadic, upper_text
from shared_observable.comparator_sampler import build as emit, configuration, summary, classical_seed_output
from shared_observable.boolean_circuit import classical_apply
from shared_observable.readout_and_comparison import exact_weights, represented_weights

PI_UP = Fraction('3.14159265358979323846264338327950288419716939937511')
RAW_MARGIN = Fraction('.002')
RAW_BIAS_MAX = Fraction('.0002')
LOGICAL_RESERVE = Fraction('.01')
FAIL_MAX = Fraction('.05')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def fractional_sum(items):
    return sum(items, Fraction(0))


def median_failure(k):
    return sum(Fraction(math.comb(k, j)) * Fraction(19, 100) ** j * Fraction(81, 100) ** (k-j)
               for j in range(k//2+1, k+1))


def finite_error(z, p):
    return 2*z*(PI_UP/p + PI_UP*PI_UP/(p*p))


def fixed_data(N, parameters):
    started = time.perf_counter()
    base = control_descriptor(N, epsilon='.0001', bits=48, **parameters)
    radius = min(base['common_head_radius'], N//2)
    iv = mp.iv
    eta = iv.mpf(base['eta'])
    sine = iv.sin(iv.pi*eta)
    hp = 1/(1+iv.mpf(base['lambda'])*iv.exp(4*iv.mpf(base['tau'])))
    coeff = [iv.mpf(v) for v in base['coefficient_intervals']]
    def q(r):
        r %= N
        if r > N//2:
            r -= N
        return (-1 if r % 2 == 0 else 1)*sine/(N*iv.tan(iv.pi*(r-eta)/N))
    weights = {}
    residuals = {}
    for j in range(radius):
        for r in (-j, j+1):
            q0 = q(r)
            g = sum(coeff[m]*(q(r-m)+q(r+m)-2*(-1)**m*q0) for m in range(1, len(coeff)))
            for dest, value in ((weights, hp*q0+g), (residuals, g)):
                integer, error = round_dyadic(value, 48)
                dest[r % N] = {'address': r % N, 'signed_numerator': integer,
                              'value_interval': enc(value), 'rounding_error_upper': upper_text(error)}
    approximation = fractional_sum(Fraction(base['error'][name][1]) for name in
        ('anchored_truncation_interval', 'coefficient_alias_interval', 'omitted_residual_tail_interval'))
    return {'base': base, 'radius': radius, 'weights': weights, 'residuals': residuals,
            'approximation_bias': approximation, 'seconds': time.perf_counter()-started}


def level_descriptor(data, low, high):
    base = data['base']
    d = copy.deepcopy(base)
    rows = data['weights'] if low == 0 else data['residuals']
    d['head'] = [rows[r % base['N']] for j in range(low, high) for r in (-j, j+1)]
    shells = list(range(high.bit_length()-1, base['n']-1)) if low == 0 else []
    d['q_tail']['shell_exponents'] = shells
    d['normalization_numerator'] = str(sum(abs(x['signed_numerator']) for x in d['head']) +
        2*len(shells)*d['q_tail']['per_side_shell_mass_numerator'])
    z = Fraction(int(d['normalization_numerator']), int(d['dyadic_denominator']))
    d['normalization'] = mp.nstr(mp.mpf(z.numerator)/mp.mpf(z.denominator), 85)
    d['descriptor_entries'] = len(d['head'])+2*len(shells)
    d['compressed_selector_entries'] = len(d['head'])+len(shells)+(1 if shells else 0)
    d['common_head_radius'] = high
    dyadic = fractional_sum(Fraction(x['rounding_error_upper']) for x in d['head'])
    if shells:
        q_interval = list(map(mp.mpf, d['q_tail']['per_side_shell_mass_interval']))
        shell_value = mp.mpf(d['q_tail']['per_side_shell_mass_numerator'])/mp.mpf(d['dyadic_denominator'])
        shell_error = max(abs(shell_value-q_interval[0]), abs(shell_value-q_interval[1]))
        dyadic += Fraction(upper_text(2*len(shells)*shell_error))
    d['error'] = {'all_label_raw_bias_upper': str(dyadic), 'dyadic_rounding_raw_upper': str(dyadic),
                  'scope': 'Only this level rounding; approximation truncation, alias and final tail counted once globally.'}
    d['multilevel'] = {'low_j': low, 'high_j_exclusive': high,
                       'kind': 'coarse heat head and cardinal tail' if low == 0 else 'smooth residual annulus',
                       'fixed_degree': base['coefficient_degree']}
    return d


def quick_counts(bundle):
    c = Counter(op[0] for op in bundle['controlled_iterate'])
    return {'T_count': 7*c['ccx']+c['t']+c['tdg'],
            'nonoracle_elementary_gates': sum(c.values())+14*c['ccx']-c['label'],
            'width': bundle['circuit'].width}


def qft_cost(m):
    # Exactly the control schedule convention, separately verified by parent checker.
    t = 3*(m-1)
    gates = 3*(m-1)
    error = Fraction(0)
    sources = []
    for j in range(3, m+1):
        path = QFT_ROOT/f'rotation_{j:02d}.json.gz'
        compressed = path.read_bytes()
        value = json.loads(gzip.decompress(compressed))
        uses = 3*(m-j+1)
        t += uses*value['T_count']
        gates += uses*len(value['word'])
        error += uses*Fraction(value['interval_operator_upper'])
        sources.append({'path': str(path), 'sha256': hashlib.sha256(compressed).hexdigest(),
                        'j': j, 'uses_per_run': uses, 'operator_error_upper': value['interval_operator_upper'],
                        'T_count_per_word': value['T_count']})
    gates += 2*m+2*(m*(m-1)//2)+3*(m//2)
    return {'T_count': t, 'nonoracle_elementary_gates': gates, 'operator_error': error, 'sources': sources}


QFT = {}


def optimize_schedule(levels, bias, query_cap=None):
    """MILP menu optimization; certify the chosen menu entry with exact rationals.

    Menu: P=2**m, 1<=m<=17; K odd, 1<=K<=41. Every nonempty level
    executes K*(P-1)>=1 times. Integer menu is explicit; no universal optimum.
    """
    L = len(levels)
    options = []
    slack = RAW_MARGIN-bias
    for i, level in enumerate(levels):
        z = Fraction(level['descriptor']['normalization'])
        for m in range(1, 18):
            p = 1 << m
            e = finite_error(z, p)
            if e >= slack:
                continue
            if m not in QFT: QFT[m] = qft_cost(m)
            qft = QFT[m]
            for k in range(1, 42, 2):
                f = median_failure(k)+2*k*qft['operator_error']
                if f >= FAIL_MAX-LOGICAL_RESERVE:
                    continue
                options.append({'level': i, 'P': p, 'm': m, 'K': k, 'error': e, 'failure': f,
                    'T': k*((p-1)*level['counts']['T_count']+qft['T_count']), 'queries': k*(p-1)})
    nconstraints = L+2+(query_cap is not None)
    matrix = lil_matrix((nconstraints, len(options)))
    for j, option in enumerate(options):
        matrix[option['level'], j] = 1
        matrix[L, j] = float(option['error'])*1e6
        matrix[L+1, j] = float(option['failure'])*1e3
        if query_cap is not None:
            matrix[L+2, j] = option['queries']
    lower = [1]*L+[0, 0]+([0] if query_cap is not None else [])
    # Tiny guard prevents a floating solver feasibility tolerance from consuming the contract.
    upper = [1]*L+[(float(slack)-1e-10)*1e6, (float(FAIL_MAX-LOGICAL_RESERVE)-1e-10)*1e3]+([query_cap] if query_cap is not None else [])
    started = time.perf_counter()
    result = milp(np.array([x['T'] for x in options])/1e9, integrality=np.ones(len(options)),
        bounds=Bounds(0, 1), constraints=LinearConstraint(matrix.tocsc(), lower, upper),
        options={'time_limit': 10, 'mip_rel_gap': 1e-7})
    if result.x is None:
        return None
    selected = [o for o, x in zip(options, result.x) if x > .5]
    assert len(selected) == L and len({o['level'] for o in selected}) == L
    selected.sort(key=lambda o:o['level'])
    total_error = fractional_sum(o['error'] for o in selected)
    total_failure = fractional_sum(o['failure'] for o in selected)+LOGICAL_RESERVE
    assert total_error+bias < RAW_MARGIN
    assert bias <= RAW_BIAS_MAX and total_failure <= FAIL_MAX
    rows = []
    total_gates = 0
    max_width = 0
    for o, level in zip(selected, levels):
        qft = QFT[o['m']]
        gates = o['queries']*level['counts']['nonoracle_elementary_gates'] + o['K']*(
            qft['nonoracle_elementary_gates']+len(level['bundle']['seedbits']))
        total_gates += gates
        max_width = max(max_width, level['counts']['width']+o['m'])
        rows.append({'level': o['level'], 'P': o['P'], 'm': o['m'], 'median_repetitions': o['K'],
            'label_queries': o['queries'], 'raw_statistical_error_fraction': str(o['error']),
            'median_failure_fraction': str(median_failure(o['K'])),
            'qft_operator_error_fraction': str(o['K']*qft['operator_error']),
            'failure_with_qft_fraction': str(o['failure']), 'T_count': o['T'],
            'nonoracle_elementary_gates': gates, 'logical_qubits': level['counts']['width']+o['m'],
            'phase_measurement_bits': o['K']*o['m'],
            'zero_initializations_upper': o['K']*(level['counts']['width']+o['m']),
            'qft_T_count': o['K']*qft['T_count'], 'qft_sources': qft['sources']})
    return {'levels': rows, 'T_count': sum(o['T'] for o in selected),
            'nonoracle_elementary_gates': total_gates, 'label_queries': sum(o['queries'] for o in selected),
            'logical_qubits_sequential_reuse': max_width, 'raw_bias_fraction': str(bias),
            'raw_statistical_error_fraction': str(total_error), 'total_raw_error_fraction': str(total_error+bias),
            'total_failure_fraction': str(total_failure), 'logical_failure_reserve_fraction': str(LOGICAL_RESERVE),
            'raw_bias': float(bias), 'total_raw_error': float(total_error+bias), 'total_failure': float(total_failure),
            'optimization': {'solver': 'scipy.optimize.milp/HiGHS', 'mip_gap': float(result.mip_gap),
                'message': result.message, 'seconds': time.perf_counter()-started, 'P_exponents': [1,17],
                'odd_K_bounds': [1,41], 'query_cap': query_cap,
                'scope': 'Menu optimum within reported solver gap, not unrestricted algorithm optimality.'}}


def make_level(desc, config):
    bundle = emit(desc, config)
    return {'descriptor': desc, 'configuration': config, 'bundle': bundle, 'counts': quick_counts(bundle)}


def coarse_configurations(desc, remaining):
    """Small registered precision menu: prune only by a stated proxy before emission."""
    z = Fraction(desc['normalization'])
    entries = desc['descriptor_entries']
    n = desc['n']
    candidates = []
    for c in range(19, 28):
        for C in range(10, 23):
            for D in range(6, 12):
                cfg = configuration(desc, '.0002', selection_bits=c, cut=C, degree=D)
                bias = Fraction(cfg['raw_bias_upper_fraction'])
                if bias < remaining:
                    # Leading literal comparator/reflection counts; this is ONLY a screening proxy.
                    proxy = 168*(n+3)*(C+2*D)+112*entries*c+112*D*c
                    candidates.append((proxy, float(bias), cfg))
    candidates.sort(key=lambda x:(x[0],x[1]))
    # Include precision tradeoffs as well as the smallest proxy costs.
    chosen = candidates[:4]
    for budget in (remaining/2, remaining/4):
        valid = [x for x in candidates if x[1] < float(budget)]
        if valid:
            chosen.append(valid[0])
    auto = configuration(desc, '.00001')
    chosen.append((0, auto['raw_bias_upper'], auto))
    seen = set()
    for _, _, cfg in chosen:
        key = tuple(cfg[x] for x in ('selection_bits','reciprocal_cut','cotangent_degree'))
        if key not in seen:
            seen.add(key)
            yield cfg


def save_bundle(path, level):
    b = level['bundle']
    value = {key:b[key] for key in ('B','A','controlled_A','controlled_phase','controlled_reflection',
        'seedbits','descriptor','configuration','categorical_intervals','cotangent_cdf','phase_target','outer_control')}
    value['B_inverse'] = list(reversed(b['B']))
    value['A_inverse'] = list(reversed(b['A']))
    value.update(width=b['circuit'].width, registers=b['circuit'].regs)
    raw = (json.dumps(value, separators=(',',':'))+'\n').encode()
    compressed = gzip.compress(raw, mtime=0)
    path.write_bytes(compressed)
    return {'path':str(path), 'sha256':hashlib.sha256(compressed).hexdigest(),
            'compressed_bytes':len(compressed), 'uncompressed_bytes':len(raw)}


def semantic_samples(level, count=20):
    b = level['bundle']
    rng = random.Random(971+level['descriptor']['multilevel']['low_j'])
    labels = [1 if rng.getrandbits(1) else -1 for _ in range(level['descriptor']['N'])]
    checks = []
    for _ in range(count):
        seed = sum(rng.getrandbits(1)<<bit for bit in b['seedbits'])
        out,_ = classical_apply(b['B'], seed)
        regs = b['circuit'].regs
        get = lambda name: sum(((out>>wire)&1)<<i for i,wire in enumerate(regs[name]))
        actual = (get('address'), get('public_sign'), get('accept'))
        expected = classical_seed_output(b, seed, labels)
        assert actual == expected[:3]
        back,_ = classical_apply(list(reversed(b['B'])), out)
        assert back == seed
        checks.append({'seed':str(seed), 'address':actual[0], 'sign':actual[1], 'accept':actual[2]})
    return {'samples':checks, 'labels':labels, 'passed':True,
            'scope':'Actual emitted Boolean compute and inverse, compared against same-seed high-level model; no large statevector claim.'}


def classical_sampler(level, rng, labels, observation=None):
    """Same finite public law, using lazy independent seed bits and CDF bisection."""
    b = level['bundle']; d = level['descriptor']; cfg=level['configuration']
    c=cfg['selection_bits']; intervals=b['categorical_intervals']
    if '_classical_ends' not in b: b['_classical_ends']=[x[1] for x in intervals]
    index=bisect.bisect_right(b['_classical_ends'], rng.getrandbits(c))
    heads=d['head']
    if index<len(heads):
        row=heads[index]
        sign=int(row['signed_numerator']<0)
        if observation is not None: observation.update(address=row['address'],public_sign=sign,accepted=True)
        return (1-2*sign)*labels[row['address']]
    k=d['q_tail']['shell_exponents'][(index-len(heads))//2];side=(index-len(heads))%2
    j=(1<<k)+rng.getrandbits(k); eta=Fraction(d['eta']);t=Fraction(j)+(1-eta if side else eta)
    bits=d['n']+eta.denominator.bit_length()-1
    frac=int((t/(1<<k)-1)*(1<<bits));u=int(2*t/d['N']*(1<<bits))
    degree=0
    while degree<=cfg['reciprocal_cut'] and rng.getrandbits(1): degree+=1
    accepted=degree<=cfg['reciprocal_cut'] and all(rng.getrandbits(bits)>=frac for _ in range(degree))
    ds=rng.getrandbits(c)
    for item in b['cotangent_cdf']:
        if item['low']<=ds<item['high']:
            accepted=accepted and not all(rng.getrandbits(bits)<u for _ in range(2*item['degree']))
            break
    address=(-j if side==0 else j+1)%d['N'];sign=(address&1)^side
    if observation is not None: observation.update(address=address,public_sign=sign,accepted=bool(accepted))
    if not accepted: return 1-2*rng.getrandbits(1)
    return (1-2*sign)*labels[address]


def classical_comparison(levels, schedule, out):
    """Matched constructive upper bounds, plus resident explicit-array timings."""
    zs=np.array([float(Fraction(x['descriptor']['normalization'])) for x in levels])
    slack=float(RAW_MARGIN-Fraction(schedule['raw_bias_fraction']))
    L=len(levels)
    # Independently estimated levels; union-Hoeffding. Optimize error/failure shares.
    def objective(v, weights):
        e=slack*np.exp(v[:L]-np.max(v[:L]));e/=sum(e)/slack
        f=.04*np.exp(v[L:]-np.max(v[L:]));f/=sum(f)/.04
        n=2*zs*zs*np.log(2/f)/(e*e)
        return float(sum(n*weights))
    result=minimize(lambda v:objective(v,np.ones(L)),np.zeros(2*L),method='BFGS')
    v=result.x;e=np.exp(v[:L]-max(v[:L]));e*=slack*.999999/sum(e)
    f=np.exp(v[L:]-max(v[L:]));f*=.04*.999999/sum(f)
    ns=np.ceil(2*zs*zs*np.log(2/f)/(e*e)).astype(np.int64)
    rows=[{'level':i,'raw_error_allocation':float(e[i]),'failure_allocation':float(f[i]),
           'samples_at_least_one':int(max(1,ns[i]))} for i in range(L)]
    weights=[]
    for level in levels:
        b=level['bundle']
        weights.append(represented_weights({'descriptor':b['descriptor'],'configuration':b['configuration'],
            'categorical_intervals':b['categorical_intervals'],'cotangent_cdf':b['cotangent_cdf']}))
    combined=sum(weights)
    rng=np.random.default_rng(628195)
    labels=rng.choice([-1.,1.],size=len(combined))
    raw=[];observations=[]
    for rep in range(64):
        start=time.perf_counter_ns();value=sum(float(np.dot(w,labels)) for w in weights)
        raw.append({'method':'same_levels_exact_dot','repeat':rep,'elapsed_ns':time.perf_counter_ns()-start})
        start=time.perf_counter_ns();value2=float(np.dot(combined,labels))
        raw.append({'method':'merged_same_levels_exact_dot','repeat':rep,'elapsed_ns':time.perf_counter_ns()-start})
        assert abs(value-value2)<1e-12
    for i,level in enumerate(levels):
        for rep in range(128):
            seed=172900000+i*1000+rep;py_rng=random.Random(seed)
            observation={'level':i,'repeat':rep,'prng_seed':seed}
            start=time.perf_counter_ns();value=classical_sampler(level,py_rng,labels,observation)
            raw.append({'method':f'same_law_level_{i}_single_sample','repeat':rep,'elapsed_ns':time.perf_counter_ns()-start})
            observation['signed_output']=int(value);observations.append(observation)
    summary_t={name:{'p50_ns':float(np.median([r['elapsed_ns'] for r in raw if r['method']==name])),
                     'p95_ns':float(np.percentile([r['elapsed_ns'] for r in raw if r['method']==name],95))}
               for name in sorted(set(r['method'] for r in raw))}
    import csv
    with (out/'classical_raw_timings.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['method','repeat','elapsed_ns']);writer.writeheader();writer.writerows(raw)
    dump(out/'classical_sample_observations.json',{'label_rng_seed':628195,'labels':labels.tolist(),
        'python_random_algorithm':'CPython random.Random.getrandbits; fresh listed seed per observation',
        'observations':observations,'scope':'Timing observations of individual samples, not repeated estimator failures or stopping curves.'})
    # Dot exactly evaluates the same finite represented law, within ordinary float roundoff.
    d=levels[0]['descriptor'];target=exact_weights(d['N'],float(d['tau']),float(d['lambda']),float(d['eta']))
    l1=float(sum(abs(target-combined)))
    assert l1<schedule['raw_bias']
    return {'same_levels':True,'same_finite_public_laws':True,'same_label_oracle':True,
        'classical_sampling':{'method':'independent per-level sample means with union-Hoeffding upper bound',
            'levels':rows,'total_samples':sum(r['samples_at_least_one'] for r in rows),
            'failure_budget':.04,'scope':'Constructive sufficient count, not a classical lower bound or optimal sampler.'},
        'deterministic_same_levels':{'distinct_label_queries':d['N'],'runtime':summary_t,
            'scope':'Resident NumPy float arrays, reused labels and public preprocessing; no hardware quantum timing claimed.'},
        'numeric_all_label_L1_to_exact_heat':l1, 'represented_weights':combined.tolist()}


def run(name, N, parameters, discovery=True, fixed_coarse=None, fixed_precision=None):
    out=ROOT/'results/unified/multilevel'/name;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();data=fixed_data(N,parameters)
    dump(out/'fixed_coefficient_descriptor.json',data['base'])
    all_ann={}
    for low in [1<<j for j in range(data['radius'].bit_length()-1)]:
        d=level_descriptor(data,low,2*low);cfg=configuration(d,'.00001')
        all_ann[low]=make_level(d,cfg)
    coarse_menu=[1,2,4,8,16,32,64] if discovery else [fixed_coarse]
    coarse_menu=[r for r in coarse_menu if r<data['radius']]
    trials=[];best=None;best_levels=None
    for radius in coarse_menu:
        ann=[all_ann[1<<j] for j in range(radius.bit_length()-1,data['radius'].bit_length()-1)]
        fixed_bias=data['approximation_bias']+fractional_sum(Fraction(x['configuration']['raw_bias_upper_fraction']) for x in ann)
        desc=level_descriptor(data,0,radius)
        if discovery:
            configs=list(coarse_configurations(desc,RAW_BIAS_MAX-fixed_bias))
        else:
            c,C,D=fixed_precision
            cfg=configuration(desc,'.0002',selection_bits=c,cut=C,degree=D)
            # Frozen confirmation policy: if necessary, tighten whichever single
            # parameter yields the largest provable raw-bias reduction. Ties c,C,D.
            while fixed_bias+Fraction(cfg['raw_bias_upper_fraction'])>RAW_BIAS_MAX:
                options=[(c+1,C,D),(c,C+1,D),(c,C,D+1)]
                alternatives=[configuration(desc,'.0002',selection_bits=x,cut=y,degree=z) for x,y,z in options]
                index=min(range(3),key=lambda j:Fraction(alternatives[j]['raw_bias_upper_fraction']))
                c,C,D=options[index];cfg=alternatives[index]
            configs=[cfg]
        for cfg in configs:
            total_bias=fixed_bias+Fraction(cfg['raw_bias_upper_fraction'])
            if total_bias>RAW_BIAS_MAX:continue
            coarse=make_level(desc,cfg);levels=[coarse]+ann
            sched=optimize_schedule(levels,total_bias)
            if sched is None:continue
            row={'coarse_radius':radius,'selection_bits':cfg['selection_bits'],'reciprocal_cut':cfg['reciprocal_cut'],
                'cotangent_degree':cfg['cotangent_degree'],'coarse_iterate_T':coarse['counts']['T_count'],
                **{k:sched[k] for k in ('T_count','label_queries','logical_qubits_sequential_reuse','raw_bias','total_raw_error','total_failure')}}
            trials.append(row)
            print(json.dumps({'trial':row}),flush=True)
            if best is None or sched['T_count']<best['T_count']:
                best=sched;best_levels=levels;best['coarse_radius']=radius
                best['coarse_precision']=[cfg['selection_bits'],cfg['reciprocal_cut'],cfg['cotangent_degree']]
    if best is None:raise RuntimeError('No valid schedule in frozen candidate menu')
    dump(out/'search_trials.json',trials)
    # Persist only the winning literal words, with full ledger and inverse checks.
    level_reports=[]
    for i,level in enumerate(best_levels):
        started=time.perf_counter();report=summary(level['bundle'])
        report['semantics']=semantic_samples(level)
        report['word']=save_bundle(out/f'level_{i:02d}_word.json.gz',level)
        report['level_descriptor']=level['descriptor']
        report['ledger_and_verification_seconds']=time.perf_counter()-started
        level_reports.append(report);dump(out/f'level_{i:02d}_ledger.json',report)
        assert report['controlled_iterate']['T_count']==level['counts']['T_count']
    classical=classical_comparison(best_levels,best,out)
    dump(out/'classical_comparison.json',classical)
    best.update({'name':name,'N':N,'parameters':parameters,'final_radius':data['radius'],
        'fixed_coefficient_degree':data['base']['coefficient_degree'], 'approximation_bias_fraction':str(data['approximation_bias']),
        'preprocessing':{'fixed_interval_descriptor_seconds':data['seconds'],
            'all_generation_search_emission_and_verification_seconds':time.perf_counter()-start,
            'selected_all_level_entries':sum(x['descriptor']['descriptor_entries'] for x in best_levels),
            'selected_all_word_compressed_bytes':sum(x['word']['compressed_bytes'] for x in level_reports),
            'selected_all_word_uncompressed_bytes':sum(x['word']['uncompressed_bytes'] for x in level_reports)},
        'classical_sampling_queries':classical['classical_sampling']['total_samples'],
        'numeric_all_label_L1_to_exact_heat':classical['numeric_all_label_L1_to_exact_heat'],
        'scope':'Literal gate words and finite rational schedule; 0.01 logical fault reserve is conditional. Unknown XOR implementation cost excluded.',
        'fine_preparation':'Annulus-only residual circuits execute separately; no full R_L head circuit occurs in the coarse stage.',
        'space_scope':'Maximum stage width plus its phase register, reusing device between measured stages.'})
    dump(out/'schedule.json',best)
    print(json.dumps({'winner':{k:best[k] for k in ('name','coarse_radius','coarse_precision','T_count','label_queries',
          'logical_qubits_sequential_reuse','raw_bias','total_raw_error','total_failure')}}),flush=True)
    return best


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['discovery','confirmation'],default='discovery')
    a=parser.parse_args()
    if a.case=='discovery':run('discovery_n12',4096,{'tau':'1','lam':'.1','eta':'.375'})
    else:
        frozen=json.loads((ROOT/'results/unified/multilevel/confirmation_registration.json').read_text())
        run('confirmation_n10',1024,{'tau':'.8','lam':'.12','eta':'.3125'},False,
            frozen['coarse_radius'],frozen['coarse_precision'])
