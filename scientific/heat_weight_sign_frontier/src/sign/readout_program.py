"""Executable finite QPE specification and literal native inverse-QFT library.

The loop counts are explicit circuit repetitions, not a claim of hardware execution.
The accepted controlled-iterate word is referenced by hash and is never recompiled.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import gzip,hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
COORD=ROOT.parent/'heat_weight_coordinate_frontier'
QFT=ROOT.parent/'heat_weight_tight_frontier/results/next/qft'

def read(path):
    p=Path(path);b=p.read_bytes();return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inverse(word):
    inv={'t':'tdg','tdg':'t','s':'sdg','sdg':'s'}
    return [(inv.get(o[0],o[0]),*o[1:]) for o in reversed(word)]

def half_rotation(j,q):
    if j==2:return [('t',q)]
    return [(g,q) for g in read(QFT/f'rotation_{j:02d}.json.gz')['word']]

def inverse_qft(m):
    forward=[]
    for j in range(m-1,-1,-1):
        forward.append(('h',j))
        for k in range(j-1,-1,-1):
            a=half_rotation(j-k+1,k);b=half_rotation(j-k+1,j)
            forward+=a+b+[('cx',k,j)]+inverse(b)+[('cx',k,j)]
    for j in range(m//2):forward += [('cx',j,m-1-j),('cx',m-1-j,j),('cx',j,m-1-j)]
    return inverse(forward)

def apply(word,state):
    a=np.asarray(state,dtype=complex).copy();idx=np.arange(len(a))
    gates={'h':np.array([[1,1],[1,-1]])/math.sqrt(2),'t':np.diag([1,np.exp(1j*math.pi/4)]),
       'tdg':np.diag([1,np.exp(-1j*math.pi/4)]),'s':np.diag([1,1j]),'sdg':np.diag([1,-1j]),
       'x':np.array([[0,1],[1,0]]),'z':np.diag([1,-1])}
    for o in word:
        if o[0]=='cx':
            _,c,t=o;keep=idx[((idx>>c)&1).astype(bool)&~((idx>>t)&1).astype(bool)]
            other=keep^(1<<t);a[keep],a[other]=a[other].copy(),a[keep].copy()
        else:
            g=gates[o[0]];q=o[1];lo=idx[(idx&(1<<q))==0];hi=lo|(1<<q)
            x=a[lo].copy();y=a[hi].copy();a[lo]=g[0,0]*x+g[0,1]*y;a[hi]=g[1,0]*x+g[1,1]*y
    return a

def depths(word,width):
    d=[0]*width;t=[0]*width
    for o in word:
        q=o[1:];a=max(d[i] for i in q)+1;b=max(t[i] for i in q)+(o[0] in ('t','tdg'))
        for i in q:d[i]=a;t[i]=b
    return dict(depth=max(d),T_depth=max(t))

def emit_qft(m):
    w=inverse_qft(m);count=Counter(o[0] for o in w)
    obj=dict(m=m,word=w,T_count=count['t']+count['tdg'],gates=len(w),**depths(w,m),
      rotation_sources=[dict(j=j,path=str(QFT/f'rotation_{j:02d}.json.gz'),sha256=digest(QFT/f'rotation_{j:02d}.json.gz')) for j in range(3,m+1)],
      scope='Inverse QFT only; initial uniform clock Hadamards are separate. Three projective half-angle Rz words per CP; their phases are global on the entire phase register.')
    p=ROOT/'results/sign'/f'inverse_qft_m{m}.json.gz'
    p.write_bytes(gzip.compress(json.dumps(obj,separators=(',',':')).encode(),mtime=0));return obj

def semantic_controls():
    rows=[]
    for m in (3,4,5,6):
        P=1<<m;word=inverse_qft(m);worst=0.;normerr=0.
        phases=[0.,math.pi,2*math.pi*(1-1/P),math.pi/2,math.pi/2+.001,math.pi/2-.001,math.pi/P,-math.pi/P]
        for kind in ('uniform','sine'):
            t=np.arange(P);amp=np.ones(P)/math.sqrt(P) if kind=='uniform' else np.sqrt(2/P)*np.sin(math.pi*(t+.5)/P)
            for phi in phases:
                inp=amp*np.exp(1j*t*phi);actual=apply(word,inp);expected=np.fft.fft(inp)/math.sqrt(P)
                phase=np.vdot(expected,actual);actual*=np.exp(-1j*np.angle(phase))
                err=np.linalg.norm(actual-expected);worst=max(worst,err)
                restored=apply(inverse(word),apply(word,inp));normerr=max(normerr,np.linalg.norm(restored-inp))
        # Stored rotation errors are ~1e-6; these numeric tests only falsify conventions.
        assert worst<1e-4 and normerr<1e-11
        rows.append(dict(m=m,cases=2*len(phases),native_QFT_state_error=worst,native_inverse_error=normerr))
    return dict(status='passed',rows=rows,scope='Small-register actual native inverse-QFT and adjoint; eigenphase distributions include +/- branches, periodic edges and half-bin removable limits. Uniform failure is established separately by proof/certificates.')

if __name__=='__main__':
    r=semantic_controls();(ROOT/'results/sign/qft_semantics.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
