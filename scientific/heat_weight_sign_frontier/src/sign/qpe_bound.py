"""Uniform finite-QPE sign certificate using exact rational outward bounds.

The proof is in results/sign/qpe_proof.md.  Floating point is only used for
human-readable summaries and for an explicitly non-certifying scout.
"""
from __future__ import annotations
from fractions import Fraction as F
from math import ceil, comb, factorial, isqrt
import argparse, hashlib, gzip, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PI_LO = F('3.14159265358979323846264338327950288419716939937510')
PI_HI = F('3.14159265358979323846264338327950288419716939937511')
PREC = 64
SCALE = 1 << PREC


def down(x, bits=PREC):
    x = F(x); q = 1 << bits
    return F(x.numerator*q // x.denominator, q)


def up(x, bits=PREC):
    x = F(x); q = 1 << bits
    return F(-((-x.numerator*q) // x.denominator), q)


def sqrt_lower(x, bits=PREC):
    x = F(x); q = 1 << bits
    return F(isqrt((x.numerator*q*q)//x.denominator), q)


def sin2_interval_upper(lo, hi):
    """Upper bound sin(pi*f)^2 for every f in closed [lo,hi] in [0,1]."""
    u = min(hi, 1-lo, F(1, 2))
    if u == F(1, 2):
        return F(1)
    # sin(z)'s degree-17 alternating partial sum is an upper bound on
    # [0, pi/2]. Enclose positive and negative terms using opposite pi ends.
    total = F(0)
    for k in range(9):
        z = (PI_HI if k % 2 == 0 else PI_LO)*u
        total += (-1)**k * z**(2*k+1)/factorial(2*k+1)
    return min(F(1), up(total*total))


def reciprocal_square_tail_upper(x, terms=24):
    """sum_{k>=0} (x+k)^-2, with an exact midpoint-integral remainder."""
    assert x > 0 and terms >= 1
    # Convexity implies sum_{k>=M} f(k) <= integral_{M-1/2}^infty f(t)dt.
    return sum((up(1/(x+k)**2) for k in range(terms)), F(0)) + up(1/(x+terms-F(1,2)))


def angular_cutoffs(P, gap):
    assert P >= 2 and P & (P-1) == 0 and 0 < gap < 2
    d = down(P*gap/(2*PI_HI), 48)
    b = down(P*sqrt_lower(2*gap)/(2*PI_HI), 48)
    assert d > 0 and b > 0
    return d, b


def interval_bound(lo, hi, d, b, terms=24):
    # The exact event uses ceil(f+d) and ceil(b-f). Endpoint-inclusive
    # relaxations cover event switches, even when a switch is an endpoint.
    forward = max(d, F(ceil(lo+d))-hi)
    backward = max(b, F(ceil(b-hi))+lo)
    tails = reciprocal_square_tail_upper(forward, terms) + reciprocal_square_tail_upper(backward, terms)
    return min(F(1), up(sin2_interval_upper(lo, hi)*tails/(PI_LO*PI_LO)))


def certificate(P, gap, target=None, tolerance=F(1,100000), terms=24, max_depth=30):
    """Exact branch-and-bound cover of all fractional phases.

    With target, prove a prescribed rational upper bound or return failure.
    Otherwise scout a proposed target plus tolerance, and prove or reject it.
    Point values are used ONLY to propose a target; correctness depends
    solely on terminal upper bounds covering the entire phase interval.
    """
    d, b = angular_cutoffs(P, gap)
    if target is None:
        if d <= F(1,4):
            # A universal probability cap is sufficient to exclude this
            # schedule under this certificate; do not spend work sharpening
            # a near-one bound that cannot amplify with a majority.
            target = F(1)
        # This is only a scheduling choice, never a bound used as a proof.
        try:
            import numpy as np
            from scipy.special import polygamma
            if target is None:
                switches=[float(F(ceil(d))-d),float(b-F(b.numerator//b.denominator))]
                probes=[x+offset for x in switches for offset in (-1e-10,0,1e-10) if 0<=x+offset<=1]
                f = np.concatenate((np.linspace(0, 1, 10001),np.asarray(probes)))
                a = np.ceil(f+float(d))-f
                c = np.ceil(float(b)-f)+f
                scout = float(np.max(np.sin(np.pi*f)**2/np.pi**2*(polygamma(1,a)+polygamma(1,c))))
                target = min(F(1),F(ceil((scout+float(tolerance))*1000000), 1000000))
        except ImportError:
            target = F(1)
    target = F(target)
    stack = [(F(0), F(1), 0)]
    leaves = []
    largest = F(0)
    while stack:
        lo, hi, depth = stack.pop()
        bound = interval_bound(lo, hi, d, b, terms)
        if bound <= target:
            leaves.append((lo,hi,bound)); largest=max(largest,bound)
        elif depth >= max_depth:
            return dict(certified=False, P=P, target=str(target), unresolved=[str(lo),str(hi)],
                        unresolved_upper=str(bound), gap=str(gap), d=str(d), b=str(b))
        else:
            mid = (lo+hi)/2
            stack.extend([(mid,hi,depth+1),(lo,mid,depth+1)])
    leaves.sort()
    assert leaves[0][0] == 0 and leaves[-1][1] == 1
    assert all(a[1] == b0[0] for a,b0 in zip(leaves,leaves[1:]))
    cover='\n'.join(','.join(map(str,row)) for row in leaves)+'\n'
    return dict(certified=True, P=P, gap=str(gap), d=str(d), b=str(b),
                p_upper=str(target), maximum_leaf_upper=str(largest),
                leaf_count=len(leaves), max_depth=max(x.denominator.bit_length()-1 for row in leaves for x in row[:2]),
                series_explicit_terms=terms, cover_sha256=hashlib.sha256(cover.encode()).hexdigest(),
                certificate_scope='Every phase, both eigenphase signs, both cosine error directions, inclusive ties; exact rational interval cover.',
                interval_cover=[[str(v) for v in row] for row in leaves])


def majority_failure(K, p):
    assert K % 2 == 1 and 0 <= p <= 1
    return sum((F(comb(K,j))*p**j*(1-p)**(K-j) for j in range(K//2+1,K+1)), F(0))


def qft_cost(m):
    """Existing schedule and exact stored rotation error receipts, read-only."""
    t=3*(m-1); gates=3*(m-1); error=F(0); sources=[]
    root=ROOT.parent/'heat_weight_tight_frontier/results/next/qft'
    for j in range(3,m+1):
        path=root/f'rotation_{j:02d}.json.gz'; blob=path.read_bytes()
        row=json.loads(gzip.decompress(blob)); uses=3*(m-j+1)
        t += uses*row['T_count']; gates += uses*len(row['word'])
        error += uses*F(row['interval_operator_upper'])
        sources.append(dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest(),uses=uses))
    gates += 2*m+m*(m-1)+3*(m//2)
    return dict(T_count=t,nonoracle_elementary_gates=gates,operator_error=error,sources=sources)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',default='discovery_n68')
    parser.add_argument('--powers',default=','.join(map(str,range(1,19))))
    parser.add_argument('--method',default='nodes'); parser.add_argument('--tolerance',default='.00003')
    parser.add_argument('--artifact',help='Accepted coordinate row filename stem; default tag_method_tapered')
    args=parser.parse_args()
    accepted=ROOT.parent/'heat_weight_coordinate_frontier/results/coordinate'
    artifact=args.artifact or f'{args.tag}_{args.method}_tapered'
    row=json.loads((accepted/f'{artifact}.json').read_text())
    Z=F(row['Z']); beta=F(row['bias']); gap=(F('.002')-beta)/Z-F(1,1<<48)
    results=[]; out=ROOT/'results/sign';out.mkdir(parents=True,exist_ok=True)
    attempt_path=out/f'qpe_attempts_{args.tag}_{args.method}.json'
    attempts=json.loads(attempt_path.read_text()) if attempt_path.exists() else []
    previous=out/f'qpe_schedule_{args.tag}_{args.method}.json'
    if previous.exists():
        for item in json.loads(previous.read_text())['certificates']:
            prior={k:v for k,v in item.items() if k not in ('interval_cover','schedules','qft_receipts')}
            if prior not in attempts: attempts.append(prior)
    for m in map(int,args.powers.split(',')):
        P=1<<m; cert=certificate(P,gap,tolerance=F(args.tolerance))
        attempt={k:v for k,v in cert.items() if k!='interval_cover'}
        if attempt not in attempts:attempts.append(attempt)
        attempt_path.write_text(json.dumps(attempts,indent=2)+'\n')
        print(json.dumps({k:v for k,v in cert.items() if k!='interval_cover'}),flush=True)
        if not cert['certified']:
            results.append(cert);continue
        (out/f'qpe_cover_{args.tag}_{args.method}_p{P}.json').write_text(json.dumps(cert,indent=2)+'\n')
        qft=qft_cost(m); p=F(cert['p_upper']); schedules=[]
        if p < F(1,2):
            for K in range(1,128,2):
                stat=majority_failure(K,p); synthesis=2*K*qft['operator_error']
                failure=stat+synthesis+F('.01')
                if failure < F('.05'):
                    baseline=row['totals'];stage=baseline['stages'][0]
                    init=baseline.get('initial_source_per_run',{}).get('T',0)
                    schedules.append(dict(P=P,K=K,p_upper=str(p),statistical_failure=str(stat),
                      qft_operator_error=str(qft['operator_error']),synthesis_failure=str(synthesis),
                      total_failure=str(failure),label_queries=baseline['measured_head_queries']+K*(P-1),
                      coherent_queries=K*(P-1),T_count=K*((P-1)*stage['iterate_T']+qft['T_count']+init)))
                    break
        record={k:v for k,v in cert.items() if k!='interval_cover'}
        record.update(schedules=schedules,qft_receipts=qft['sources']); results.append(record)
    options=[s for r in results for s in r.get('schedules',[])]
    result=dict(tag=args.tag,method=args.method,Z=str(Z),beta=str(beta),gap=str(gap),
      accepted_artifact=artifact,
      baseline=row['totals'],certificates=results,best=min(options,key=lambda s:s['T_count']) if options else None,
      menu=dict(phase_powers=list(map(int,args.powers.split(','))),odd_repetitions=list(range(1,128,2)),
      scope='Minimum complete T among this declared menu certified by this bound, not global readout optimality.',
      domination='For a fixed P, failure below .05 first occurs at saved K, and every larger K has larger full T.'))
    (out/f'qpe_schedule_{args.tag}_{args.method}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BEST',json.dumps(result['best']),flush=True)


if __name__=='__main__':
    main()
