"""Independent exact validation of random-offset sine raw certificates.

Imports no producer modules. Trigonometric endpoint ceilings are checked
with independent 120-digit interval cosine matrices, not the producer's
Taylor polynomial. Remaining certificate inequalities are exact rationals.
"""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import math
import sys
import mpmath as mp

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/random_sign'
mp.iv.dps=120
LO=F('3.14159265358979323846264338327950288419716939937510')
HI=F('3.14159265358979323846264338327950288419716939937511')
A=F(127,128)


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def ceil(v):return -(-v.numerator//v.denominator)


def dyadic_upper(interval):
    sign,mantissa,exponent,_=interval._mpi_[1]
    return (-1 if sign else 1)*F(mantissa)*F(2)**exponent


def side_conditions():
    def atan(v):
        partial=sum(((-1)**k*v**(2*k+1)/F(2*k+1) for k in range(60)),F(0))
        return partial,partial+v**121/F(121)
    a,b=atan(F(1,5));c,d=atan(F(1,239))
    assert LO<16*a-4*d<16*b-4*c<HI
    # Validity of near and far trigonometric inequalities for every P>=32.
    assert 1-HI**2/F(6*16**2)>A
    assert LO/F(16)-(HI/F(16))**3/F(6)>F(3,16)
    assert HI/2<2
    assert F(9,256)-F(4,32**2)>0
    assert F(3,4)-F(1,128)>1/(2*A)


@lru_cache(None)
def cosine_interval_upper(L,U):
    if L==0 or U==1:return F(1)
    # cos^2(pi f) has one interior minimum and no interior maximum on [0,1].
    def endpoint(f):
        angle=2*mp.iv.pi*mp.iv.mpf(f.numerator)/f.denominator
        return dyadic_upper((1+mp.iv.cos(angle))/2)
    return max(endpoint(L),endpoint(U))


@lru_cache(None)
def ray_rational(v):
    assert v>1/(2*A)
    # n>=8 decreasing tail <= integral from v+7. Replace the varying
    # factor A^2-1/(4*t^2) by its smaller endpoint value before integrating.
    prefix=sum((F(1)/(A*A*(v+n)**2-F(1,4))**2 for n in range(8)),F(0))
    edge=v+7
    integral=F(1)/(3*edge**3*(A*A-F(1)/(4*edge**2))**2)
    return prefix+integral


@lru_cache(None)
def fine_ray(v):
    return F(ceil(ray_rational(v)*2**100),2**100)


@lru_cache(None)
def fine_cosine(L,U):
    return F(ceil(cosine_interval_upper(L,U)*2**80),2**80)


def review_menu():
    path=OUT/'bound_menu.json';saved=json.loads(path.read_text())
    source=Path(saved['accepted_input']);assert digest(source)==saved['accepted_input_sha256']
    accepted=json.loads(source.read_text(encoding='utf-8-sig'))
    bases={(r['tag'],r['response']):r for r in accepted['rows'] if r['route']=='sine_sign'}
    rows=saved['rows'];assert len(rows)==len(bases)*18*5
    keys={(r['tag'],r['response'],r['P'],r['S']) for r in rows}
    assert len(keys)==len(rows)
    assert keys=={(tag,response,2**m,S) for tag,response in bases for m in range(1,19) for S in (1,2,4,8,16)}
    validated=0;caps=0
    for row in rows:
        base=bases[row['tag'],row['response']]
        Z,beta=F(base['Z']),F(base['bias']);P,S=row['P'],row['S']
        gap=(F(1,500)-beta)/Z-F(1,2**48)
        assert F(row['g'])==gap and F(row['Z'])==Z and F(row['bias'])==beta
        qf,qb=F(row['q_forward']),F(row['q_backward']);root=F(row['sqrt_2g_lower'])
        assert root>0 and root**2<=2*gap
        assert qf==P*gap/(2*HI) and qb==P*root/(2*HI)
        if P<32 or qf<=F(3,4):
            assert F(row['statistical_failure'])==1;caps+=1;continue
        largest=F(0)
        for i in range(128):
            total=F(0)
            for z in range(S):
                L,U=F(i+128*z,128*S),F(i+128*z+1,128*S)
                forward=ceil(qf-U)+L;backward=ceil(qb+L)-U
                total+=fine_cosine(L,U)*(fine_ray(forward)+fine_ray(backward))/(2*LO**2)
            largest=max(largest,total/S)
        far=F(8)/(P**3*(F(9,256)-F(4,P**2))**2)
        assert min(F(1),largest+far)<=F(row['statistical_failure']), (row['tag'],row['response'],P,S)
        validated+=1
    result=dict(status='passed',path=str(path.relative_to(ROOT)),sha256=digest(path),
                entries=len(rows),nontrivial_bounds=validated,probability_caps=caps,
                scope='All registered P and S entries independently rebuilt using finer independent interval-cosine and ray rounding. No native-cost selection reviewed here.')
    (OUT/'review_menu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


def review(path,expected=None):
    path=Path(path).resolve()
    saved=json.loads(path.read_text(encoding='utf8'))
    c=saved['certificate'];P,S=c['P'],c['S']
    assert P>=32 and P&(P-1)==0 and S>=1 and S&(S-1)==0
    Z,beta=F(c['Z']),F(c['bias'])
    if expected is None:
        source=Path(saved['accepted_input'])
        assert digest(source)==saved['accepted_input_sha256']
        accepted=json.loads(source.read_text(encoding='utf-8-sig'))
        matches=[r for r in accepted['rows'] if r['tag']==saved['tag'] and r['response']==saved['response'] and r['route']=='sine_sign']
        assert len(matches)==1
        expected=(F(matches[0]['Z']),F(matches[0]['bias']))
    assert (Z,beta)==tuple(map(F,expected))
    assert 0<=beta<=F(1,5000) and Z>0
    gap=(F(1,500)-beta)/Z-F(1,2**48)
    assert gap==F(c['g']) and gap>0
    root=F(c['sqrt_2g_lower'])
    assert root>0 and root**2<=2*gap
    qf,qb=F(c['q_forward']),F(c['q_backward'])
    assert 0<qf<=P*gap/(2*HI) and 0<qb<=P*root/(2*HI)
    assert qf>F(3,4)
    rawfar=F(8)/(P**3*(F(9,256)-F(4,P**2))**2)
    far=F(c['far_upper']);assert rawfar<=far
    cells=c['cells'];assert len(cells)==128
    prior=F(0);maxcell=F(0);branch_count=0
    for i,cell in enumerate(cells):
        L,U=F(cell['L']),F(cell['U'])
        assert cell['index']==i and L==prior and U>L
        assert U-L<=F(1,128*S)
        prior=U
        branches=cell['branches'];assert len(branches)==S
        branchsum=F(0)
        for z,b in enumerate(branches):
            assert b['z']==z
            l,u=F(b['L']),F(b['U'])
            assert (l,u)==(L+F(z,S),U+F(z,S)) and 0<=l<=u<=1
            nf,nb=b['forward_integer'],b['backward_integer']
            assert nf==ceil(qf-u) and nb==ceil(qb+l)
            vf,vb=F(b['v_forward']),F(b['v_backward'])
            assert vf==nf+l and vb==nb-u
            C=F(b['cosine_squared_upper'])
            assert 0<=C<=1 and cosine_interval_upper(l,u)<=C
            rf,rb=F(b['ray_forward_upper']),F(b['ray_backward_upper'])
            assert ray_rational(vf)<=rf and ray_rational(vb)<=rb
            value=F(b['near_upper'])
            assert C*(rf+rb)/(2*LO**2)<=value
            branchsum+=value;branch_count+=1
        average=F(cell['averaged_near_upper'])
        assert branchsum/S<=average
        maxcell=max(maxcell,average)
    assert prior==F(1,S)
    maxnear=F(c['max_averaged_near_upper'])
    assert maxcell<=maxnear
    total=F(c['statistical_failure'])
    assert min(F(1),maxnear+far)<=total<=1
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path),P=P,S=S,
                cells=len(cells),branches=branch_count,statistical_upper=str(total),
                scope='Every phase; average over finite classical branches before supremum; both eigenphases and both error directions by joint-outcome symmetry.')


def main():
    side_conditions()
    paths=sorted(OUT.glob('bound_*_P*_S*.json'))
    assert paths
    rows=[review(p) for p in paths]
    result=dict(status='passed',certificate_count=len(rows),rows=rows,
                method='No producer imports; Machin bracket from 60 exact terms, independent interval cosine endpoint ceilings, exact rational unrounded ray and cover inequalities.',
                limitation='This validates the supplied statistical certificates, not native synthesis or complete program costs.')
    (OUT/'review_certificates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    if '--menu' in sys.argv:review_menu()


if __name__=='__main__':main()
