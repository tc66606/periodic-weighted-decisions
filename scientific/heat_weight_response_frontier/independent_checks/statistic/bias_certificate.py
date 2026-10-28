"""Fresh rational/interval review of saved stochastic interpolation certificates.

No producer modules are imported or executed. Standard historical heat remainder
and BHMT theorems are interfaces; every new numerical bound is rebuilt here.
Run with the bundled Python; output is results/statistic/bias_review.{json,md}.
"""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import gzip, hashlib, json, math
import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/statistic'
mp.mp.dps = 130
mp.iv.dps = 120
I = mp.iv
PI = F('3.14159265358979323846264338327950288419716939937511')
MARGIN, CAP, FAILURE, RESERVE = F('.002'), F('.0002'), F('.05'), F('.01')
INPUTS = {}
QFT_REPLAYS = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def endpoint(t):
    sign, mantissa, exponent, _ = t
    return F((-1 if sign else 1)*mantissa)*F(2)**exponent


def bounds(value):
    return tuple(endpoint(t) for t in value._mpi_)


def upper(value):
    return bounds(value)[1]


def interval_fraction(q):
    q = F(q)
    return I.mpf(q.numerator)/q.denominator


def error(value, q):
    lo, hi = bounds(value)
    return max(abs(F(q)-lo), abs(F(q)-hi))


def outward(q, places=24):
    q = F(q)
    scale = 10**places
    n = -((-q.numerator*scale)//q.denominator)
    return f'{n // scale}.{n % scale:0{places}d}'


def summary(q):
    return {'fraction': str(q), 'decimal_upper': outward(q)}


@lru_cache(None)
def coefficient_data(N, tau_text, lam_text, eta_text, degree, samples):
    tau, lam, eta = map(I.mpf, (tau_text, lam_text, eta_text))
    sine = I.sin(I.pi*eta)
    hp = 1/(1+lam*I.exp(4*tau))
    sample = [1/(1+lam*I.exp(2*tau*(1-I.cos(2*I.pi*j/samples))))
              for j in range(samples)]
    coef = [sum((sample[j]*I.cos(2*I.pi*j*m/samples)
                 for j in range(samples)), I.mpf(0))/samples
            for m in range(degree+1)]
    moment = sum((m*m*abs(coef[m]) for m in range(1, degree+1)), I.mpf(0))
    qcache = {}

    def q(r):
        r %= N
        if r > N//2:
            r -= N
        if r not in qcache:
            qcache[r] = (-1 if r % 2 == 0 else 1)*sine/(N*I.tan(I.pi*(r-eta)/N))
        return qcache[r]

    def residual(r):
        return sum((coef[m]*(q(r-m)+q(r+m)-2*(-1)**m*q(r))
                    for m in range(1, degree+1)), I.mpf(0))

    # Accepted analytic heat remainder interfaces, re-evaluated independently.
    sx = I.pi/(4*tau)
    a = I.mpf('0.9')*I.ln(sx+I.sqrt(1+sx*sx))
    z = I.exp(-a)
    c = sine/I.pi
    A = 1/eta + 1/(1-eta)
    U = 4*c*(A+2*I.ln(5)) + I.pi**2*sine
    V = 8*c
    trunc = z**(degree+1)*((U+V*(degree+1))/(1-z)+V*z/(1-z)**2)
    alias = sum((((z**(samples-m)+z**(samples+m))/(1-z**samples))
                 *(4*c*(A+2*(1+I.ln(4*m+1)))+I.pi**2*sine)
                 for m in range(1, degree+1)), I.mpf(0))

    def global_bound(radius):
        tail = I.mpf(0) if 2*radius >= N else 8*sine/(3*I.pi)*moment*sum(
            (1/(radius+off)**3+1/(2*(radius+off)**2) for off in (eta,1-eta)), I.mpf(0))
        return {'anchored_truncation': upper(trunc), 'coefficient_alias': upper(alias),
                'original_residual_tail': upper(tail)}

    return {'coef': coef, 'moment': moment, 'q': q, 'residual': residual,
            'mass': hp*c, 'hp': hp, 'global_bound': global_bound}


def context(d):
    return coefficient_data(d['N'], d['tau'], d['lambda'], d['eta'],
                            d['coefficient_degree'], d['coefficient_samples'])


def check_descriptor(d, measured=None, original_radius=None):
    ctx = context(d)
    assert d['N'] == 1 << d['n']
    den = int(d['dyadic_denominator'])
    assert den == 1 << d['dyadic_bits']
    eta = F(d['eta'])
    assert 0 < eta < 1 and eta.denominator & (eta.denominator-1) == 0
    for real, saved in zip(ctx['coef'], d['coefficient_intervals']):
        lo, hi = bounds(real)
        assert F(saved[0]) <= lo <= hi <= F(saved[1])
    moment_lo, moment_hi = bounds(ctx['moment'])
    sm = d['coefficient_second_absolute_moment_interval']
    assert F(sm[0]) <= moment_lo <= moment_hi <= F(sm[1])
    shell = d['q_tail']['shell_exponents']
    mass = F(d['q_tail']['per_side_shell_mass_numerator'], den)
    assert mass > 0
    ml, mh = bounds(ctx['mass'])
    sm = d['q_tail']['per_side_shell_mass_interval']
    assert F(sm[0]) <= ml <= mh <= F(sm[1])
    rows = d['head'] if measured is None else measured
    addresses = [r['address'] for r in rows]
    assert len(addresses) == len(set(addresses))
    dyadic_error = F(0)
    low = d['multilevel']['low_j']
    original_radius = original_radius or d['multilevel']['high_j_exclusive']
    for row in rows:
        r = row['address']
        # Original full heat head, residual annulus, or extended cardinal head.
        canonical = r if r <= d['N']//2 else r-d['N']
        in_original = -original_radius < canonical <= original_radius
        val = ctx['residual'](r) if low else ctx['hp']*ctx['q'](r)
        if not low and in_original:
            val += ctx['residual'](r)
        e = error(val, F(row['signed_numerator'],den))
        if 'rounding_error_upper' in row:
            assert e <= F(row['rounding_error_upper'])
        dyadic_error += e
    dyadic_error += 2*len(shell)*error(ctx['mass'],mass)
    declared = F(d['error']['all_label_raw_bias_upper'])
    assert dyadic_error <= declared
    z = F(sum(abs(x['signed_numerator']) for x in d['head']),den)+2*len(shell)*mass
    assert z == F(int(d['normalization_numerator']),den)
    assert d['descriptor_entries'] == len(d['head'])+2*len(shell)
    if shell:
        radius = 1 << min(shell)
        assert shell == list(range(min(shell),d['n']-1))
        if measured is not None:
            assert set(addresses) == {r%d['N'] for j in range(radius) for r in (-j,j+1)}
            assert len(rows)+2*sum(1<<k for k in shell) == d['N']
        for k in shell:
            # Both offsets give 1 <= t/L < 2, preserve exact n-bit addresses.
            assert F(1) <= 1+eta/(1<<k) < 2
            assert F(1) <= 1+(1-eta)/(1<<k) < 2
    return {'normalization': str(z), 'descriptor_rounding_recomputed_upper':summary(dyadic_error),
            'descriptor_rounding_declared': summary(declared), 'head_entries_verified':len(rows),
            'coefficient_intervals_verified':len(ctx['coef']), 'shells':len(shell)}


def check_whole(d, cfg, table, keys):
    g, p, c, v = (cfg[k] for k in ('g','p','c','round_bits'))
    M, scale = 1<<g, 1<<p
    shift = F(1,8*M*M) if cfg['centered'] else F(0)
    maximum_node = F(0)
    negatives = 0
    for key, values in table.items():
        assert len(values) == M+1
        for j, (mag, sign) in enumerate(values):
            assert 0 <= mag < scale and sign in (0,1)
            x = I.mpf(1)+I.mpf(j)/M
            if key == 'limit':
                node = 1/x-interval_fraction(shift)
            elif int(key) == d['n']-2 and j == M:
                node = -interval_fraction(shift)
            else:
                a = I.pi*(1<<int(key))/d['N']
                node = a*I.cos(a*x)/I.sin(a*x)-interval_fraction(shift)
            e = error(node,F(-mag if sign else mag,scale))
            assert e <= F(1,scale)
            maximum_node = max(maximum_node,e)
            negatives += bool(sign)
    shell = d['q_tail']['shell_exponents']
    mass = F(d['q_tail']['per_side_shell_mass_numerator'], int(d['dyadic_denominator']))
    z = F(int(d['normalization_numerator']),int(d['dyadic_denominator']))
    small = [k for k in shell if cfg.get('merge_small_cut') is not None and k<=d['n']-cfg['merge_small_cut']]
    assert set(table) == {str(keys[str(k)]) for k in shell}
    for k in shell:
        assert str(keys[str(k)]) == ('limit' if k in small else str(k))
    exact_bits = max(shell)+F(d['eta']).denominator.bit_length()-1
    assert exact_bits == cfg['exact_fraction_bits']
    bary = F(0) if g+v >= exact_bits else PI**2/F(8*(1<<(g+v)))
    masses = [abs(r['signed_numerator']) for r in d['head']]+[d['q_tail']['per_side_shell_mass_numerator']]*(2*len(shell))
    # Cumulative downward rounding independently certifies selector L1.
    previous, cumulative, l1 = 0, 0, F(0)
    total = sum(masses)
    for m in masses:
        cumulative += m
        end = cumulative*(1<<c)//total
        l1 += abs(F(end-previous,1<<c)-F(m,total))
        previous = end
    assert previous == 1<<c and l1 <= F(2*len(masses),1<<c)
    terms = {'descriptor':F(d['error']['all_label_raw_bias_upper']),
             'selector':z*F(2*len(masses),1<<c),
             'interpolation':2*len(shell)*mass*(shift if cfg['centered'] else F(1,4*M*M)),
             'node_quantization':2*len(shell)*mass/F(scale),
             'coordinate_floor':2*len(shell)*mass*bary,
             'small_shell_template':2*mass*sum((F(8*(1<<(2*k)),d['N']**2) for k in small),F(0))}
    return sum(terms.values(),F(0)), {'terms':{k:summary(x) for k,x in terms.items()},
        'maximum_independently_enclosed_node_error':summary(maximum_node),
        'actual_selector_L1':summary(l1),'table_entries_verified':sum(map(len,table.values())),
        'negative_signed_endpoints':negatives,'shell_mapping_verified':len(shell)}


def check_other_bias(d,cfg,check_declared=True):
    den = int(d['dyadic_denominator'])
    z = F(int(d['normalization_numerator']),den)
    shells = d['q_tail']['shell_exponents']
    mass = F(d['q_tail']['per_side_shell_mass_numerator'],den)
    tail = 2*len(shells)*mass
    weighted = 2*mass*sum((F(1<<(k+2),d['N']) for k in shells),F(0))
    c = cfg['selection_bits']
    bias = F(d['error']['all_label_raw_bias_upper'])+z*F(2*d['descriptor_entries'],1<<c)
    exact = d['n']+F(d['eta']).denominator.bit_length()-1
    if 'q_rec' in cfg:
        h = F(1,1<<cfg['q_cot'])
        cot = sum((2*mass*PI**2/4*(u*(h-u) if u<h/2 else h*h/4)
                   for u in (min(F(1),F(1<<(k+2),d['N'])) for k in shells)),F(0))
        bias += tail/F(4*4**cfg['q_rec']) + cot + 2*tail/F(1<<cfg['probability_bits'])
    elif shells:
        C,D = cfg['reciprocal_cut'],cfg['cotangent_degree']
        bias += z*(F(1,1<<(C+1))+F(4,3*4**D)+F(D,1<<c))
    if 'reciprocal_bits' in cfg:
        if cfg['reciprocal_bits']<exact: bias += tail/F(1<<cfg['reciprocal_bits'])
        if cfg['cotangent_bits']<exact: bias += PI**2*weighted/F(4*(1<<cfg['cotangent_bits']))
    if check_declared: assert bias == F(cfg['raw_bias_upper_fraction'])
    return bias


def check_factor_tables(cfg,tables):
    scale=1<<cfg['probability_bits']
    checked=0
    for kind,q in [('reciprocal',cfg['q_rec']),('cotangent',cfg['q_cot'])]:
        values=tables[kind]; M=1<<q
        assert len(values)==M+1
        for j,represented in enumerate(values):
            u=I.mpf(j)/M
            if kind=='reciprocal': real=1/(1+u)
            elif j==0: real=I.mpf(1)
            elif j==M: real=I.mpf(0)
            else: real=I.pi*u/2*I.cos(I.pi*u/2)/I.sin(I.pi*u/2)
            assert 0<=represented<=scale
            assert error(real,F(represented,scale))<=F(1,scale)
            checked+=1
    return {'independently_checked_factor_endpoints':checked,
            'scope':'Mathematical tables; separate circuit review checks representation of the exact probability-one endpoint.'}


@lru_cache(None)
def qft(m):
    h = 1/I.sqrt(2)
    ii = I.mpc(0,1)
    t = I.exp(ii*I.pi/4)
    matrices = {'h':I.matrix([[h,h],[h,-h]]),'t':I.matrix([[1,0],[0,t]]),
       'tdg':I.matrix([[1,0],[0,1/t]]),'s':I.matrix([[1,0],[0,ii]]),
       'sdg':I.matrix([[1,0],[0,-ii]]),'x':I.matrix([[0,1],[1,0]]),
       'z':I.matrix([[1,0],[0,-1]])}
    error_total, T = F(0),3*(m-1)
    rows=[]
    for j in range(3,m+1):
        path=ROOT.parent/f'heat_weight_tight_frontier/results/next/qft/rotation_{j:02d}.json.gz'
        d=read(path)
        U=I.eye(2)
        for gate in d['word']: U=matrices[gate]*U
        angle=I.pi/(1<<j)
        V=I.matrix([[I.exp(-ii*angle/2),0],[0,I.exp(ii*angle/2)]])
        # The exact stored binary64 phase is enclosed, not estimated anew.
        phase=I.exp(ii*I.mpf(d['comparison_global_phase']))
        E=U-phase*V
        frob=I.sqrt(sum((abs(E[a,b])**2 for a in range(2) for b in range(2)),I.mpf(0)))
        actual=upper(frob)
        saved=F(d['interval_operator_upper'])
        assert actual <= saved, (j,outward(actual),outward(saved))
        uses=3*(m-j+1)
        gates=sum(x in ('t','tdg') for x in d['word'])
        assert gates == d['T_count']
        T+=uses*gates
        error_total+=uses*saved
        rows.append({'j':j,'uses':uses,'independent_interval_Frobenius_upper':summary(actual),
                     'saved_upper':summary(saved),'T_per_word':gates})
    result={'T':T,'error':error_total,'rotation_checks':rows}
    QFT_REPLAYS[m]=result
    return result


def median(k):
    assert k>0 and k%2==1
    return sum((F(math.comb(k,j))*F(19,100)**j*F(81,100)**(k-j)
                for j in range(k//2+1,k+1)),F(0))


def statistical(z,P):
    assert P>1 and P&(P-1)==0
    return 2*z*(PI/P+PI*PI/(P*P))


def finite_decoder_witness():
    # Exact rational cos Taylor polynomial through degree64 (order65) at
    # 2*PI*j/P, then nearest 48-bit rounding. This is a sufficient classical
    # implementation, not a claim that it was emitted by the producer.
    pi_lo,_=bounds(I.pi)
    remainder=(2*PI)**66/F(math.factorial(66))+2*(PI-pi_lo)
    bound=remainder+F(1,1<<49)
    assert bound<F(1,1<<48)
    path=OUT/'finite_decoder_witness.json'
    actual=read(path)
    maximum=F(0)
    for row in actual['cases']:
        P,j=row['P'],row['j'];x=2*PI*j/P
        assert P>1 and P&(P-1)==0 and 0<=j<P
        # Different evaluation arrangement from the producer's recurrence.
        poly=sum(((-1)**r*x**(2*r)/F(math.factorial(2*r)) for r in range(33)),F(0))
        got=F(row['signed_mean_fraction'])
        assert (got*(1<<48)).denominator==1
        assert abs(got-poly)<=F(1,1<<49)
        e=error(I.cos(2*I.pi*j/P),got)
        assert e<F(1,1<<48)
        maximum=max(maximum,e)
    source=ROOT/'src/statistic/finite_decoder.py'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    INPUTS[str(source)]=source_hash
    return {'method':'For every integer phase outcome j, evaluate the rational cosine Taylor polynomial sum r=0..32 at 2*PI_UP*j/P; round to nearest multiple of 2^-48.',
            'uniform_cosine_error_upper':summary(bound),'charged_signed_mean_error':str(F(1,1<<48)),
            'actual_decoder_outputs_verified':len(actual['cases']),'maximum_witness_error':summary(maximum),'source_sha256':source_hash,
            'scope':'Supplied finite classical decoder; source read, saved outputs independently checked, no producer import. No extra label calls or quantum T gates; classical arithmetic is separate from emitted quantum counts.'}


def readout(z,bias,s,head=0,iterate_T=None):
    P,k=s['P'],s.get('k',s.get('K'))
    ft=qft(P.bit_length()-1)
    e=statistical(z,P)
    failure=median(k)+2*k*ft['error']+RESERVE
    queries=head+k*(P-1)
    assert bias<CAP and bias+e<MARGIN and failure<FAILURE
    decoder=z/F(1<<48)
    assert bias+e+decoder<MARGIN
    assert abs(float(bias+e)-s['raw_error'])<1e-17
    assert abs(float(failure)-s['failure'])<1e-17
    assert k*(P-1)==s['quantum_queries']
    T=k*((P-1)*iterate_T+ft['T']) if iterate_T is not None else None
    if T is not None: assert T==s['T_count']
    return {'P':P,'K':k,'bias':summary(bias),'BHMT_raw_error':summary(e),
       'total_raw_error':summary(bias+e),'raw_slack':summary(MARGIN-bias-e),
       'optional_48bit_decoder_raw_error':summary(decoder),
       'total_with_optional_finite_decoder':summary(bias+e+decoder),
       'median_failure':summary(median(k)),'QFT_distribution_perturbation':summary(2*k*ft['error']),
       'conditional_logical_reserve':str(RESERVE),'total_failure':summary(failure),
       'quantum_queries':k*(P-1),'head_queries':head,'total_queries':queries,'T_from_count_interface':T}


def global_check(d,radius,declared):
    terms=context(d)['global_bound'](radius)
    total=sum(terms.values(),F(0))
    assert total<=F(declared)
    return {'terms':{k:summary(v) for k,v in terms.items()},'recomputed_upper':summary(total),
            'declared_upper':summary(F(declared))}


def main_point():
    raw=read(OUT/'blind_high_precision_n68_R2048_raw.json.gz')
    report=read(OUT/'blind_high_precision_n68_R2048.json')
    fixed=F(read(OUT/'blind_compiled_hybrid_n68_R2048.json')['fixed_coefficient_and_original_tail_bias'])
    d=raw['descriptor']
    desc=check_descriptor(d,raw['head_coefficients'],128)
    b,detail=check_whole(d,raw['configuration'],raw['endpoint_table'],raw['table_keys'])
    assert fixed+b == F(raw['raw_bias_fraction'])
    glob=global_check(d,128,fixed)
    complete=readout(F(raw['normalization_fraction']),fixed+b,report['schedule'],
                     len(raw['head_coefficients']),report['counts']['T_count'])
    assert complete['total_queries']==report['total_label_queries']==118777
    historical=read(ROOT.parent/'heat_weight_tight_frontier/results/next/finite_query_certificates.json')
    acceptance=read(ROOT.parent/'heat_weight_tight_frontier/results/next/fresh_theory_review.json')
    assert acceptance['claims']['new_n68_certificate']['verdict']=='ACCEPT'
    found=[]
    def walk(value):
        if isinstance(value,dict):
            if value.get('n')==68 and value.get('classical_integer_lower')==472711:found.append(value)
            for x in value.values():walk(x)
        elif isinstance(value,list):
            for x in value:walk(x)
    walk(historical)
    assert found
    old=found[0]
    for name,want in [('tau','1'),('lambda','.1'),('eta','.375'),('gamma','.002'),('delta','.05')]:
        assert F(old[name])==F(want)
    assert 118777<472711
    return {'verdict':'ACCEPT_WITH_RESTRICTION','descriptor':desc,'global_heat_approximation':glob,
      'sampler':detail,'readout':complete,'classical_lower_interface':472711,
      'query_ratio_classical_lower_over_new_upper':summary(F(472711,118777)),
      'restriction':'Unknown n-bit XOR labels and the previously accepted exact-heat lower-bound interface; conditional logical failure reserve. No explicit-array runtime or hardware speedup follows.'}


def multilevel(path,group,result):
    schedule=read(path/group/'schedule.json')
    bias=F(result['fixed_bias_fraction']); err=F(0); decoder=F(0); failure=RESERVE; queries=0; Ts=0; details=[]
    glob=None
    for row in schedule['levels']:
        dsc=read(path/group/f'level{row["level"]}_description.json')
        d,cfg=dsc['descriptor'],dsc['configuration']
        detail=check_descriptor(d)
        if glob is None:glob=global_check(d,result['fresh_descriptor_radius'],result['fixed_bias_fraction'])
        if 'g' in cfg:
            b,interp=check_whole(d,cfg,dsc['table'],dsc['table_keys'])
            detail['whole_product']=interp
        else:
            b=check_other_bias(d,cfg)
            if 'q_rec' in cfg: detail['factor_tables']=check_factor_tables(cfg,dsc['factory_tables'])
        assert b==F(row['raw_bias_fraction'])
        z=F(int(d['normalization_numerator']),int(d['dyadic_denominator']))
        P,k=row['P'],row['K'];ft=qft(P.bit_length()-1)
        e=statistical(z,P);f=median(k)+2*k*ft['error']
        assert e==F(row['statistical_error_fraction'])
        assert f==F(row['failure_including_qft_fraction'])
        assert k*ft['T']==row['QFT_T']
        assert k*(P-1)==row['label_queries']
        assert k*((P-1)*row['iterate_T']+ft['T'])==row['complete_T']
        bias+=b;err+=e;decoder+=z/F(1<<48);failure+=f;queries+=k*(P-1);Ts+=row['complete_T']
        details.append({'level':row['level'],'descriptor':detail,'bias':summary(b),'statistical_error':summary(e),
                        'failure_including_qft':summary(f),'P':P,'K':k})
    assert bias<CAP and bias+err<MARGIN and failure<FAILURE
    assert bias+err+decoder<MARGIN
    assert queries==schedule['label_queries'] and Ts==schedule['T_count']
    assert bias==F(schedule['raw_bias_fraction'])
    assert bias+err==F(schedule['total_raw_error_fraction'])
    assert failure==F(schedule['total_failure_fraction'])
    return {'verdict':'ACCEPT','global_heat_approximation':glob,'levels':details,
            'bias':summary(bias),'total_raw_error':summary(bias+err),
            'optional_48bit_decoder_raw_error':summary(decoder),
            'total_with_optional_finite_decoder':summary(bias+err+decoder),
            'failure':summary(failure),'queries':queries,'T_from_count_interface':Ts}


def confirmation(path):
    result=read(path/'result.json')
    groups={}
    for name,s in result['schedules'].items():
        if s['status']=='unresolved':
            groups[name]={'verdict':'UNRESOLVED_AS_REPORTED','reason':s['reason']}
        else:groups[name]=multilevel(path,name,result)
    raw=read(path/'hybrid_description.json');d=raw['descriptor']
    descriptor=check_descriptor(d,raw['head_coefficients'],result['fresh_descriptor_radius'])
    fixed=F(result['fixed_bias_fraction'])
    global_result=global_check(d,result['fresh_descriptor_radius'],fixed)
    b,detail=check_whole(d,raw['configuration'],raw['endpoint_table'],raw['table_keys'])
    assert fixed+b==F(raw['raw_bias_fraction'])
    hybrid=[]
    for row in result['hybrid']['methods']:
        # Whole-product bias is independently rebuilt; old methods remain controls.
        bias=fixed+b if row['kind']=='whole_product' else F(row['raw_bias_fraction'])
        r=readout(F(row['normalization_fraction']),bias,row['schedule'],len(raw['head_coefficients']),row['compiled_ledger']['T_count'])
        assert r['total_queries']==row['total_label_queries']
        hybrid.append({'kind':row['kind'],'readout':r,'bias_scope':'reconstructed' if row['kind']=='whole_product' else 'saved control interface; readout reconstructed'})
    rejection_check=check_rejections(path,result) if any(v['status']=='unresolved' for v in result['schedules'].values()) else None
    return {'case':result['case'],'multilevel':groups,'hybrid':{'verdict':'ACCEPT_WITH_RESTRICTION',
            'global_heat_approximation':global_result,'descriptor':descriptor,'sampler':detail,'methods':hybrid,
            'restriction':'Exact N-query classical algorithm remains cheaper in label calls at these explicit small N.'},
            'rejected_menu_rows':len(result['rejected']),'independent_infeasibility_check':rejection_check}


def check_rejections(path,result):
    frozen=read(ROOT/'configs/statistic/finalist_rules_frozen.json')['whole_product_rule']
    d=read(path/'strong_control/level0_description.json')['descriptor']
    mass=F(d['q_tail']['per_side_shell_mass_numerator'],int(d['dyadic_denominator']))
    z=F(int(d['normalization_numerator']),int(d['dyadic_denominator']))
    shell=d['q_tail']['shell_exponents']; tail=2*len(shell)*mass
    fixed=F(result['fixed_bias_fraction']); declared={x.get('name'):x for x in result['rejected']}
    rows=[]
    for c,g,p,center in frozen['finalist1_coarse_menu']:
        b=F(d['error']['all_label_raw_bias_upper'])+z*F(2*d['descriptor_entries'],1<<c)
        b+=tail*(F(1,(8 if center else 4)*(1<<(2*g)))+F(1,1<<p))
        b+=2*mass*sum((F(8*(1<<(2*k)),d['N']**2) for k in shell if k<=d['n']-8),F(0))
        name=f'whole_c{c}_g{g}_p{p}_center{int(center)}'
        assert b==F(declared[name]['raw_bias_fraction']) and fixed+b>=CAP
        rows.append({'id':name,'fixed_plus_coarse_bias':summary(fixed+b)})
    exact=d['n']+F(d['eta']).denominator.bit_length()-1
    shared=[]
    for qr in (6,7):
        for qc in (6,7):
            for p in (16,17):
                for c in (21,22):
                    for retain in (False,True):
                        cfg={'selection_bits':c,'q_rec':qr,'q_cot':qc,'probability_bits':p,
                             'reciprocal_bits':min(exact,16),'cotangent_bits':min(exact,16)}
                        b=check_other_bias(d,cfg,False)
                        name=f'shared_{qr}_{qc}_{p}_{c}_{retain}'
                        assert b==F(declared[name]['bias_fraction']) and fixed+b>=CAP
                        shared.append({'id':name,'fixed_plus_coarse_bias':summary(fixed+b)})
    return {'whole_product_rejections':rows,'shared_factors_rejections':shared,
            'conclusion':'All five whole-product and 32 shared-factor frozen coarse choices exceed the total bias cap before adding fine-level errors. P/K changes cannot repair this.'}


def uncentered_endpoint_counterexample():
    # Static representability counterexample to the generic producer domain.
    # No producer import or execution: a p-bit XOR load keeps only the low p bits.
    N,eta,k,g,p=8192,F(5,16),5,6,18
    bary=(1<<g)*eta/(1<<k)
    stored=1<<p
    represented=stored&((1<<p)-1)
    assert bary==F(5,8) and represented==0
    loss=(1-bary)*(F(stored,1<<p)-F(represented,1<<p))
    assert loss==F(3,8)
    return {'verdict':'PREPATCH_REVISE_CONSTRUCTOR_DOMAIN','configuration':{'N':N,'eta':str(eta),'k':k,'g':g,'p':p,'centered':False,'merge_small_cut':8},
            'offset':0,'side':0,'lower_endpoint_probability':str(1-bary),
            'table_magnitude':stored,'loaded_p_bit_magnitude':represented,'signed_response_magnitude_loss':str(loss),
            'finding':'The uncentered shared 1/x endpoint at x=1 is exactly one. The producer allocates only p threshold bits and emits its lower p bits; the value 2^p therefore becomes zero. At this address endpoint zero is reachable with probability 3/8.',
            'finite_scope':'Both affected N8192 uncentered frozen rows were already rejected by their bias caps. All accepted whole-product points reviewed here are centered. At N4096 g=6 the merged endpoint zero is unreachable, so this counterexample does not invalidate that discovery point.',
            'required_before_generic_claim':'Handle exact probability one explicitly, widen the threshold encoding, or restrict the constructor claim to centered configurations / proven-unreachable unit endpoints. Preserve the frozen confirmation results.'}


def check_correction_chain():
    path=OUT/'corrections/unit_endpoint_correction.json'
    if not path.exists():return {'verdict':'PENDING','reason':'No correction receipt saved yet.'}
    receipt=read(path)
    frozen=read(ROOT/'configs/statistic/finalist_rules_frozen.json')
    archive=ROOT/receipt['archived_source']
    revised=ROOT/receipt['frozen_source']
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(archive)==receipt['frozen_sha256']==frozen['source_hashes'][receipt['frozen_source']]
    assert sha(revised)==receipt['corrected_sha256']
    for name,want in frozen['source_hashes'].items():
        if name!=receipt['frozen_source']: assert sha(ROOT/name)==want
    assert sha(ROOT/'configs/statistic/confirmation.json')==frozen['confirmation_registration_sha256']
    selected=[]
    for row in receipt['selected_programs_unchanged']:
        p=ROOT/row['raw_path'];n=ROOT/row['native_path']
        assert sha(p)==row['raw_sha256'] and sha(n)==row['native_sha256']
        saved=read(p);raw=saved.get('raw',saved)
        assert 'unit_probability_threshold' not in raw['registers']
        if 'configuration' in raw: assert raw['configuration']['centered']
        table=raw.get('endpoint_table') or raw.get('table')
        if table:
            scale=1<<raw['configuration']['p']
            assert all(mag<scale for values in table.values() for mag,_ in values)
        selected.append({'raw_path':row['raw_path'],'hash_verified':True,'unit_flag_absent':True})
    def apply(word,x):
        for name,*w in word:
            if name=='x': x^=1<<w[0]
            elif name=='cx': x^=((x>>w[0])&1)<<w[1]
            elif name=='ccx': x^=(((x>>w[0])&(x>>w[1]))&1)<<w[2]
            else: raise AssertionError('Unexpected non-Boolean compute gate '+name)
        return x
    checks=[]
    for version in ('before','after'):
        raw=read(OUT/f'corrections/unit_endpoint_{version}_raw.json.gz')
        reg=raw['registers']
        for item in receipt['cases']:
            if item['version']!=version:continue
            seed=int(item['seed']);got=apply(raw['B'],seed)
            accept=(got>>reg['accept'][0])&1
            endpoint_value=sum(((got>>q)&1)<<i for i,q in enumerate(reg['endpoint_index']))
            threshold_value=sum(((got>>q)&1)<<i for i,q in enumerate(reg['accept_threshold']))
            assert endpoint_value==0 and threshold_value==0
            assert accept==(version=='after')
            if version=='after': assert (got>>reg['unit_probability_threshold'][0])&1
            assert apply(list(reversed(raw['B'])),got)==seed
            checks.append({'version':version,'coin':item['coin'],'accept':accept,'inverse_restores_seed':True})
    return {'verdict':'ACCEPT_DOCUMENTED_CORRECTION','frozen_source_sha256':receipt['frozen_sha256'],
            'corrected_source_sha256':receipt['corrected_sha256'],'selected_hash_checks':selected,'independent_B_replays':checks,
            'proof':'The extra unit flag is set only at a one-hot selected endpoint of magnitude 2^p; its low-p-bit threshold is zero, so threshold comparison and unit flag are mutually exclusive. XOR equals OR and acceptance is identically one. For centered selected tables every magnitude is below 2^p, so all newly added conditional emitter branches are inactive and allocate no flag. The remaining frozen source and confirmation registration hashes are unchanged.',
            'scope':'Correctness-only source repair after frozen confirmation; no new accuracy/menu/schedule tuning. Original rejected N8192 choices remain rejected.'}


def main():
    assert upper(I.pi)<PI
    assert 8/PI**2>F(81,100)
    point=main_point()
    cases=[confirmation(p) for p in sorted((OUT/'confirmation').iterdir()) if p.is_dir() and (p/'result.json').exists()]
    report={'reviewer':'fresh independent mathematical and finite-bias review','producer_imports':False,
      'arithmetic':'120-decimal outward mpmath.iv; exact binary endpoint to Fraction conversion; rational final inequalities.',
      'verdict':'ACCEPT_WITH_RESTRICTION','main_n68':point,'confirmation':cases,
      'uncentered_constructor_counterexample':uncentered_endpoint_counterexample(),
      'documented_endpoint_correction':check_correction_chain(),
      'finite_classical_decoder':finite_decoder_witness(),
      'proof_checks':{
        'curvature':'For 0<a<=pi/4 and 1<=x<=2, f_a prime=-a^2 csc^2(ax), f_a double-prime=2a^3 csc^2(ax)cot(ax)>=0. The convergent positive cotangent series gives f_a double-prime<=2/x^3<=2; a/sin(a) is increasing, so |f_a prime|<=pi^2/8.',
        'centering':'Convex interpolation has error in [0,h^2/4]; subtracting h^2/8 centers it in [-h^2/8,h^2/8]. Signed endpoints are retained. Independent rounding and acceptance seeds give a linear signed expectation, including negative endpoints.',
        'template':'Partial fractions make (1-z cot(z))/z^2 increasing up to pi/2, with endpoint 4/pi^2. Thus 0<=1/x-a cot(ax)<=8(L/N)^2. Geometric summation gives (64/3)m_sh*4^-c, and code sums actual shells exactly.',
        'sign_and_shell':'Side0 r=-j, sign bit=r mod2; side1 r=j+1, sign bit=1 XOR(r mod2). This equals sign of cardinal coefficient. Distances j+eta and j+1-eta cover the complement of the measured head exactly, without rounding addresses.',
        'selector':'Cumulative floor boundaries give each endpoint discrepancy <2^-c. Their adjacent differences give L1<=2E*2^-c. Selector bias is Z times this L1, charged once; head rows have exact response sign.',
        'readout':'BHMT success >=8/pi^2>0.81 implies independent-run failure <=0.19. Signed conversion multiplies probability error by 2Z. Median binomial tail plus 2K times QFT operator error plus the 0.01 reserve bounds total worst-input failure.',
        'head':'Measured labels make c_H exact for the dyadic vector. Every promised input obeys |true score-(c_H+estimated tail)|<=bias+statistical error<gamma. No observed score or tail label is used for schedule selection.'},
      'restrictions':['The accepted historical exact-heat remainder and classical lower-bound theorems are interfaces, not re-proved here.',
        'The 0.01 logical reserve is a conditional implementation budget; no physical noise model or fault-tolerant implementation is supplied.',
        'Native sampler circuit semantics and literal counts are a separate independent review; T here uses their emitted-count interface.',
        'N=8192 frozen standalone whole-product/shared-factor menus remain unresolved; successful hybrid confirmation does not validate those standalone menus.',
        'N=2^68 is an oracle separation. Explicit arrays have input/storage cost; no end-to-end runtime, GPU, hardware, or universal finite-menu optimality claim is accepted.',
        'This is internal separate-context machine review, not external human peer review.'],
      'qft_interval_replays':{str(m):{'T':v['T'],'error':summary(v['error']),'rotation_checks':v['rotation_checks']}
                              for m,v in sorted(QFT_REPLAYS.items())},
      'input_sha256':INPUTS}
    (OUT/'bias_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    r=point['readout']
    text=f'''# Independent mathematical and finite-bias review

**ACCEPT_WITH_RESTRICTION.** This review imports no producer code. It independently evaluates saved coefficients, head weights, signed endpoint tables and QFT words with 120-digit outward interval arithmetic, converts binary interval endpoints exactly to rational numbers, and reconstructs final budgets with `Fraction`.

The N = 2^68 whole-product/measured-head point has raw bias at most {r['bias']['decimal_upper']}, original total raw error at most {r['total_raw_error']['decimal_upper']}, and worst-input failure at most {r['total_failure']['decimal_upper']} < 0.05, including the conditional 0.01 logical reserve. Charging the supplied finite classical decoder gives complete numerical raw error at most **{r['total_with_optional_finite_decoder']['decimal_upper']} < 0.002**. It uses 7(16384-1) = 114681 coherent label calls plus 4096 measured head calls, totaling **118777**, below the accepted same-parameter exact-heat classical lower-bound interface **472711**. The raw interval table contains 4617 verified signed endpoints. T accounting from the separately checked iterate-count interface gives 18810668940.

For f_a(x)=a cot(ax), direct derivatives give nonnegative curvature; its convergent positive cotangent series bounds curvature by 2/x^3. Convex interpolation error lies in [0,h^2/4]. Subtracting h^2/8 gives the claimed centered bound h^2/8; the negative endpoint is valid only as a signed response. Independent endpoint-rounding and acceptance seeds give its exact linear expectation. The slope bound pi^2/8 pays for coordinate flooring. The shared 1/x substitution follows from the cotangent partial fractions and contributes at most 8(L/N)^2 per shell response. The checker charges descriptor rounding, selector L1, interpolation, node quantization, coordinate rounding and shell substitution separately and exactly once.

The sign convention and shell map match the cardinal tail. The measured head and tail partition all addresses. Extending the head retains the original omitted-residual certificate; it does not claim new exact heat coefficients. Measured labels enter only the final shifted decision threshold. Thus bias plus statistical error below gamma proves the sign for every promised input, not only sampled labels.

BHMT success 8/pi^2 exceeds 0.81. Independent median repetition gives the exact binomial tail. Every used QFT synthesis word is independently interval-multiplied; its operator error is bounded by the Frobenius norm and lies below the saved bound. The sum of run distribution perturbations is at most 2K times the QFT operator error. Head measurement and public numerical approximation add no probabilistic failure.

The original schedules implicitly used exact classical phase decoding. The supplied `src/statistic/finite_decoder.py` closes this numerical interface: it evaluates the rational degree64 cosine Taylor polynomial using the certified rational pi upper bound, then rounds to 48 fractional bits. Taylor's theorem and the pi interval give signed-mean error less than 2^-48, so adding Z*2^-48 per level preserves every strict raw-margin inequality. All 61 saved outputs, including quadrant boundaries and adjacent phase points across nine phase sizes, independently match the rational rounding rule and outward cosine intervals. This decoder adds no label queries or quantum T gates; its classical arithmetic is separate from the emitted quantum counts.

Fresh N=2048 and N=8192 measured-head hybrid candidates both pass the rational bias/readout checks. The N=2048 standalone whole-product schedule also passes. The N=8192 frozen standalone whole-product and shared-factor menus are unresolved as reported; this is a finite-menu feasibility failure, and must remain visible. At both explicit small N, querying all N labels is cheaper in label calls than these coherent schedules.

**The generic uncentered constructor required and received a documented correction.** With uncentered shared 1/x tables, the endpoint at x=1 has magnitude 2^p, but the prepatch p-bit threshold register loaded only its lower p bits, yielding zero. A concrete reachable case is N=8192, eta=5/16, shell k=5, g=6, p=18, side0, offset0: the lower endpoint is selected with probability 3/8, so the signed response lost magnitude 3/8. Both affected N8192 frozen rows were already rejected on bias cap; all accepted whole-product points checked here use centering. For N4096 with g=6 the merged endpoint zero is unreachable. The archived prepatch source matches the frozen hash; the separate correction receipt matches the revised source. Its optional probability-one predicate repairs this case, and independent Boolean replay verifies both extreme acceptance seeds and clean inversion. All ten selected raw/native artifact hashes match the receipt, and centered tables activate none of the new emitter branches. Frozen confirmation is preserved, including rejected rows.

Restrictions: the established heat remainder and classical lower-bound proofs are accepted interfaces; this review does not redo the historical query theory. Native Boolean/Clifford+T semantics and literal resource counts belong to the separate circuit review. The 0.01 logical reserve is conditional, with no physical noise or fault-tolerant construction. The N=2^68 result is an unknown-XOR oracle separation, with no explicit-array runtime, hardware, GPU, or universal optimum conclusion. This is internal separate-context review, not human peer review.

The executable review is `independent_checks/statistic/bias_certificate.py`; exact fractions, component budgets, confirmations and input hashes are saved in `results/statistic/bias_review.json`.
'''
    (OUT/'bias_review.md').write_text(text,encoding='utf-8')
    print(json.dumps({'verdict':report['verdict'],'total_queries':118777,'main_raw_error':r['total_raw_error']['decimal_upper'],
                      'main_failure':r['total_failure']['decimal_upper'],'confirmation_cases':[x['case']['name'] for x in cases]}))


if __name__=='__main__':
    main()
