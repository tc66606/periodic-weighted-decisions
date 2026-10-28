"""Independent one-run sine-clock sign decision; accepted inputs are read-only.

Run --native to emit certified clock words and complete scalar ledgers.
Analytic guarantees use exact rational arithmetic; sampled distributions only
check the implementation of the formula and are explicitly not certificates.
"""
from pathlib import Path
from fractions import Fraction as F
import sys
sys.dont_write_bytecode = True
import argparse, gzip, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ACCEPTED = ROOT.parent / 'heat_weight_coordinate_frontier'
OUT = ROOT / 'results/sign'
PI_LO = F('3.14159265358979323846264338327950288419716939937510')
PI_HI = F('3.14159265358979323846264338327950288419716939937511')
GAMMA, RESERVE, FAIL = F('.002'), F('.01'), F('.05')
_COS_BINS = None


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, default=str) + '\n'
    path = OUT / name
    if name.endswith('.gz'):
        with gzip.open(path, 'wt', encoding='utf8') as f:
            f.write(data)
    else:
        path.write_text(data, encoding='utf8')


def tail_bound(P, q):
    """Both circular rays, every eigenphase, all shifted thresholds."""
    a = F(31, 32)
    if P < 32 or q <= 1/(2*a):
        return F(1)
    return min(F(1), (q**-4 + 1/(3*q**3))/(PI_LO**2*(a*a-1/(4*q*q))**2) + F(800,P**3))


def ceiling(value):
    return -((-value.numerator)//value.denominator)


def cosine_bins():
    """Outward dyadic bounds on max cos^2(pi*t) on 128 rational cells."""
    global _COS_BINS
    if _COS_BINS is not None:return _COS_BINS
    import mpmath as mp
    mp.iv.dps=80
    out=[]
    for i in range(128):
        lo,hi=F(i,128),F(i+1,128)
        endpoint=lo if hi<=F(1,2) else hi
        angle=2*mp.iv.pi*endpoint.numerator/endpoint.denominator
        value=(1+mp.iv.cos(angle))/2
        sign,man,exp,bc=value._mpi_[1]
        upper=F((-1)**sign*man)*(F(2)**exp)
        rounded=min(F(1),F(ceiling(upper*2**64),2**64))
        out.append((lo,hi,rounded))
    _COS_BINS=out
    return out


def refined_tail_bound(P,q):
    """Exact rational interval envelope; avoids discarding the common cosine.

    Frozen menu: 128 phase-fraction cells, 8 explicit ray terms, near region
    d<=P/16, a=127/128. There is no label or empirical-margin tuning.
    """
    if P<32 or q<=F(3,4):return F(1)
    a=F(127,128)
    def ray(d):
        assert d>1/(2*a)
        direct=sum((a*a*(d+n)**2-F(1,4))**-2 for n in range(8))
        D=d+7
        remainder=1/(3*D**3*(a*a-1/(4*D*D))**2)
        return direct+remainder
    near=F(0)
    for lo,hi,cosmax in cosine_bins():
        dp=ceiling(q-hi)+lo
        dm=ceiling(q+lo)-hi
        near=max(near,cosmax*(ray(dp)+ray(dm))/(2*PI_LO**2))
    far=8/(F(P)**3*(F(9,256)-F(4,P**2))**2)
    return min(tail_bound(P,q),near+far,F(1))


def certificate(P, Z, bias, clock_error, qft_error):
    h = (GAMMA-bias-Z/F(2**48))/Z
    q = P*h/(2*PI_HI)
    original = tail_bound(P,q)
    stat = refined_tail_bound(P,q)
    total = stat + 2*clock_error + 2*qft_error + RESERVE
    return dict(P=P, q=str(q), q_float=float(q), statistical_failure=str(stat),
                initial_conservative_failure=str(original),initial_conservative_failure_float=float(original),
                statistical_failure_float=float(stat), clock_error=str(clock_error),
                qft_error=str(qft_error), total_failure=str(total),
                total_failure_float=float(total), feasible=total<FAIL)


def phase_probabilities(P, offset):
    d = (offset-np.arange(P)+P/2) % P - P/2
    z = np.pi*d/P
    with np.errstate(divide='ignore', invalid='ignore'):
        p = (2/P**2 * np.cos(np.pi*d)**2 * np.cos(z)**2 * np.sin(np.pi/(2*P))**2
             / (np.sin(z)**2-np.sin(np.pi/(2*P))**2)**2)
    p[np.isclose(np.abs(d), .5, rtol=0, atol=1e-12)] = .5
    return d,p


def ideal_tests():
    rows=[]
    for m in (2,3,5,11,15):
        P=1<<m
        x=np.arange(P//2)
        angle=np.pi-np.pi/P-2*np.pi*x/P
        prepared=np.concatenate([np.cos(angle/2),np.sin(angle/2)])/np.sqrt(P/2)
        desired=np.sqrt(2/P)*np.sin(np.pi*(np.arange(P)+.5)/P)
        error=float(np.max(np.abs(prepared-desired)))
        assert error<1e-14
        formula_error=0.
        for offset in (0.,.1,.499,.5,.8):
            _,p=phase_probabilities(P,offset)
            amplitudes=np.fft.fft(desired*np.exp(2j*np.pi*offset*np.arange(P)/P))/np.sqrt(P)
            formula_error=max(formula_error,float(np.max(np.abs(p-np.abs(amplitudes)**2))))
            assert abs(float(np.sum(p))-1)<1e-10
        assert formula_error<1e-10
        rows.append(dict(m=m,clock_max_amplitude_error=error,distribution_max_error=formula_error))
    return rows


def dependencies():
    for path in [ROOT/'src',ROOT/'src/response',ROOT/'src/statistic',
                 ROOT.parent/'heat_weight_shared_observable/src',ROOT.parent/'heat_weight_unified_frontier/src']:
        sys.path.insert(0,str(path))
    import autonomous_blind_source as source
    from shared_observable.boolean_circuit import inverse, sparse_apply
    from comparison_table import literal_depth
    return source,inverse,sparse_apply,literal_depth


def native_clock(m):
    source,inverse,sparse_apply,literal_depth=dependencies()
    import mpmath as mp
    mp.mp.dps=80;mp.iv.dps=75
    P=1<<m;top=m-1
    word=[('h',q) for q in range(m-1)]
    fixed,rec=source.synthesized_ry(mp.pi*(1-mp.mpf(1)/P),mp.iv.pi*(1-mp.iv.mpf(1)/P),top)
    word+=fixed
    error=F(rec['interval_projective_error'])
    receipts=[dict(kind='fixed',**rec)]
    for k in range(m-1):
        # theta=-2*pi*2^k/P; W approximates Ry(theta/2).
        theta=-mp.pi*mp.mpf(2**k)/P
        theta_iv=-mp.iv.pi*mp.iv.mpf(2**k)/P
        W,rec=source.synthesized_ry(theta,theta_iv,top)
        # Applied left-to-right: W X W^dagger X = Ry(2*theta) when controlled.
        word += [('cx',k,top)] + inverse(W) + [('cx',k,top)] + W
        error+=2*F(rec['interval_projective_error'])
        receipts.append(dict(kind='controlled_half_angle',control=k,**rec))
    depth,tdepth=literal_depth(word,m)
    value=dict(m=m,P=P,word=word,T=sum(g[0] in ('t','tdg') for g in word),
               nonoracle_gates=len(word),T_depth=tdepth,nonoracle_depth=depth,
               width=m,operator_error=str(error),rotation_receipts=receipts)
    if m<=7:
        actual=sparse_apply(word,{0:1},None)
        wanted=np.sqrt(2/P)*np.sin(np.pi*(np.arange(P)+.5)/P)
        vector=np.array([actual.get(j,0) for j in range(P)])
        phase=np.vdot(wanted,vector);phase=phase/abs(phase)
        state_error=float(np.linalg.norm(vector-phase*wanted))
        assert state_error<float(error)+1e-12
        value['state_error_observed']=state_error
    save(f'blind_clock_m{m:02d}.json.gz',value)
    return value


def main():
    args=argparse.ArgumentParser();args.add_argument('--native',action='store_true');opts=args.parse_args()
    tests=ideal_tests()
    rows=json.loads((ACCEPTED/'results/coordinate/final_comparison.json').read_text(encoding='utf-8-sig'))['rows']
    clocks={};qfts={};result=[]
    if opts.native:
        source,*_=dependencies()
        # A small native witness separately tests signs and literal-adjoint control.
        native_clock(5)
    for row in rows:
        Z,bias=F(row['Z']),F(row['bias'])
        stage=row['totals']['stages'][0]
        for m in range(5,18):
            P=1<<m
            preliminary=certificate(P,Z,bias,F(1,10000),F(1,10000))
            if not preliminary['feasible']:continue
            if opts.native:
                if m not in clocks:clocks[m]=native_clock(m)
                if m not in qfts:qfts[m]=source.ml.qft_cost(m)
                clock,qft=clocks[m],qfts[m]
                cert=certificate(P,Z,bias,F(clock['operator_error']),qft['operator_error'])
            else:
                clock=qft=None;cert=preliminary
            assert cert['feasible']
            break
        else:raise AssertionError('No feasible clock in finite menu')
        q=float(F(cert['q']));max_tail=0.;worst_offset=None
        for offset in np.linspace(0,1,1025):
            d,p=phase_probabilities(P,float(offset))
            observed=float(np.sum(p[np.abs(d)>=q]))
            if observed>max_tail:max_tail,worst_offset=observed,float(offset)
        assert max_tail<float(F(cert['statistical_failure']))
        item=dict(tag=row['tag'],method=row['method'],accepted_artifact=row['artifact'],
                  accepted_bias=row['bias'],accepted_Z=row['Z'],certificate=cert,
                  distribution_probe=dict(offset_count=1025,max_tail=max_tail,worst_offset=worst_offset,
                                          scope='Numerical regression test, not the all-phase certificate'),
                  accepted_T=row['totals']['T_count'],accepted_T_depth=row['totals']['T_depth_upper'],
                  coherent_queries=P-1,total_label_queries=P-1+row['totals']['measured_head_queries'],
                  dominant_T=(P-1)*stage['iterate_T'],dominant_T_depth=(P-1)*stage['iterate_T_depth'],
                  allocated_width=stage['iterate_width']+m)
        if opts.native:
            initial=row['totals']['initial_source_per_run']
            item.update(T_count=item['dominant_T']+initial['T']+clock['T']+qft['T_count'],
                        T_depth_upper=item['dominant_T_depth']+initial['T_depth']+clock['T_depth']+qft['T_count'],
                        nonoracle_depth_upper=(P-1)*stage['iterate_nonoracle_depth']+initial['nonoracle_depth']+clock['nonoracle_depth']+qft['nonoracle_elementary_gates'],
                        clock_T=clock['T'],clock_T_depth=clock['T_depth'],QFT_T=qft['T_count'],
                        clock_word=f'blind_clock_m{m:02d}.json.gz',qft_receipts=qft['sources'])
        result.append(item)
        print(json.dumps({k:item[k] for k in ('tag','method','coherent_queries','dominant_T','allocated_width')},default=str),flush=True)
    report=dict(mechanism='one sine-clock QPE, same G and observable',ideal_tests=tests,rows=result,
                refined_bound_constants=dict(phase_cells=128,explicit_ray_terms=8,near_divisor=16,
                                             a='127/128',far_sine_lower='3/16',cosine_uppers=[str(x[2]) for x in cosine_bins()]),
                certificate='Analytic exact rational two-sided tail, all phases and all measured heads',
                contract=dict(raw_margin='.002',raw_bias_cap='.0002',total_failure='.05',logical_reserve='.01'),
                limitation='No physical noise realization; existing native iterate and descriptor costs inherited unchanged')
    save('blind_sine_native.json' if opts.native else 'blind_sine_probe.json',report)


if __name__=='__main__':main()
