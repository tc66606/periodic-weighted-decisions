"""Literal finite-offset phase ramps, clocks and inverse Fourier words.

Classical z is fixed within one execution, and freshly drawn in every repetition.
All branch costs are recorded; neither parallel branches nor free phase gates.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from functools import lru_cache
import gzip,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT.parent/'heat_weight_sign_frontier'
LIB=ROOT.parent/'heat_weight_tight_frontier/results/next/qft'
OUT=ROOT/'results/random_sign/native'
PI_HI=F('3.14159265358979323846264338327950288419716939937511')

def read(p):
    p=Path(p);b=p.read_bytes();return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(obj,indent=None if p.suffix=='.gz' else 2,default=str)+'\n').encode()
    p.write_bytes(gzip.compress(b,mtime=0) if p.suffix=='.gz' else b);return p
def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ACCEPTED=import_file('_native_accepted_readout',OLD/'src/sign/readout_program.py')
inverse,depths,apply=ACCEPTED.inverse,ACCEPTED.depths,ACCEPTED.apply
def ledger(word,width):
    c=Counter(o[0] for o in word)
    d=depths(word,width) if word else dict(depth=0,T_depth=0)
    return dict(T=c['t']+c['tdg'],gates=len(word),**d)
def ref(p):return dict(path=str(p),sha256=sha(p))

@lru_cache(None)
def rotation(j):
    if j==1:return ['s'],F(0),dict(j=j,kind='exact S')
    if j==2:return ['t'],F(0),dict(j=j,kind='exact T')
    if j>=21:
        e=PI_HI/F(2**(j+1))
        return [],e,dict(j=j,kind='identity approximation',operator_error=str(e),proof='||I-Rz(pi/2^j)|| <= pi/2^(j+1); no synthesis receipt assumed')
    p=LIB/f'rotation_{j:02d}.json.gz';r=read(p);assert r['accepted']
    return r['word'],F(r['interval_operator_upper']),dict(j=j,kind='accepted native Rz',**ref(p))

def rz(j,q,negative=False):
    w,e,r=rotation(j);w=[(g,q) for g in w]
    return (inverse(w) if negative else w),e,r

@lru_cache(None)
def ramps(m,S):
    assert S>0 and S&(S-1)==0
    r=S.bit_length()-1;branches=[]
    for z in range(S):
        w=[];e=F(0);terms=[]
        for k in range(m):
            for h in range(r):
                if not (z>>h)&1:continue
                j=m+r-k-h-1;v,err,receipt=rz(j,k,True)
                w+=v;e+=err
                terms.append(dict(clock_bit=k,random_bit=h,rotation_index=j,operator_error=str(err),receipt=receipt))
        branches.append(dict(z=z,word=w,inverse_word=inverse(w),operator_error=str(e),terms=terms,**ledger(w,m)))
    worst={k:max(b[k] for b in branches) for k in ('T','gates','depth','T_depth')}
    mean={k:str(sum(F(b[k]) for b in branches)/S) for k in worst}
    result=dict(schema='finite_classical_ramp_v1',m=m,P=2**m,S=S,random_bits=r,
        exponent='-2*pi*i*z*t/(S*P)',branches=branches,worst=worst,mean=mean,
        worst_operator_error=str(max(F(b['operator_error']) for b in branches)),
        mean_operator_error=str(sum(F(b['operator_error']) for b in branches)/S),
        scope='One classical alternative is executed; z freshly independent in each repetition. Branch global phases are irrelevant; no coherent superposition of z is assumed.')
    p=save(OUT/f'ramp_m{m:02d}_S{S}.json.gz',result);return result,ref(p)

@lru_cache(None)
def qft(m):
    forward=[];error=F(0);receipts={}
    for j in range(m-1,-1,-1):
        forward.append(('h',j))
        for k in range(j-1,-1,-1):
            a,e,rec=rz(j-k+1,k);b,_,_=rz(j-k+1,j)
            forward+=a+b+[('cx',k,j)]+inverse(b)+[('cx',k,j)]
            error+=3*e;receipts[j-k+1]=rec
    for j in range(m//2):forward += [('cx',j,m-1-j),('cx',m-1-j,j),('cx',j,m-1-j)]
    w=inverse(forward)
    result=dict(m=m,word=w,inverse_word=forward,operator_error=str(error),rotation_receipts=list(receipts.values()),**ledger(w,m))
    p=save(OUT/f'iqft_m{m:02d}.json.gz',result);return result,ref(p)

@lru_cache(None)
def clock(m):
    p=OLD/f'results/sign/blind_clock_m{m:02d}.json.gz'
    if not p.exists():
        p=OUT/f'blind_clock_m{m:02d}.json.gz'
        if not p.exists():
            mod=import_file('_accepted_sine_clock',OLD/'src/sign/blind_sine.py')
            mod.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True);mod.native_clock(m)
    r=read(p);return dict(word=r['word'],operator_error=r['operator_error'],**ledger(r['word'],m)),ref(p)

def signed_mean(j,P,z=0,S=1):
    if not isinstance(z,int) or not isinstance(S,int) or S<1 or S&(S-1) or not 0<=z<S:raise ValueError('Require a finite power-of-two offset and valid branch.')
    if not isinstance(j,int) or not 0<=j<P:raise ValueError('Invalid measured outcome')
    decoder=import_file('_accepted_decoder',OLD/'src/statistic/finite_decoder.py')
    return decoder.signed_mean(S*j+z,S*P,bits=48)

def decision(outcomes,P,S,Z,head=F(0)):
    """Outcomes is an odd list of (j,z), with new independent z each run."""
    if not outcomes or len(outcomes)%2!=1:raise ValueError('Odd nonempty observations required')
    values=sorted(signed_mean(j,P,z,S) for j,z in outcomes)
    return 1 if F(head)+F(Z)*values[len(values)//2]>=0 else -1

if __name__=='__main__':
    for S in (1,2,4,8,16):
        r,p=ramps(14,S);print(S,r['worst'],r['mean'],r['worst_operator_error'])
