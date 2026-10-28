"""Independent native padding, clock, IQFT and decoder verification.

The emitted artifact words are the inputs; no constructor is imported.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
import math
import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/random_sign'
ACCEPTED=ROOT.parent/'heat_weight_sign_frontier'
LIB=ROOT.parent/'heat_weight_tight_frontier/results/next/qft'
mp.iv.dps=120
PI_LO=F('3.14159265358979323846264338327950288419716939937510')
PI_HI=F('3.14159265358979323846264338327950288419716939937511')


def read(path):
    p=Path(path);data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upper(x):
    s,n,e,_=x._mpi_[1]
    return F((-1)**s*n)*F(2)**e


def inverse(word):
    conjugates={'t':'tdg','tdg':'t','s':'sdg','sdg':'s'}
    return [[conjugates.get(op[0],op[0]),*op[1:]] for op in reversed(word)]


def counts(word,width):
    depth=[0]*width;tdepth=[0]*width;c=Counter(op[0] for op in word)
    for op in word:
        wires=op[1:]
        dt=1+max(depth[q] for q in wires)
        tt=int(op[0] in ('t','tdg'))+max(tdepth[q] for q in wires)
        for q in wires:depth[q]=dt;tdepth[q]=tt
    return dict(T_count=c['t']+c['tdg'],gates=len(word),depth=max(depth),T_depth=max(tdepth))


def expected_iqft(n):
    # Direct inverse decomposition in execution order, independent of producer.
    word=[]
    for q in reversed(range(n//2)):
        k=n-1-q;word.extend([['cx',q,k],['cx',k,q],['cx',q,k]])
    for target in range(n):
        for control in range(target):
            order=target-control+1
            bare=['t'] if order==2 else read(LIB/f'rotation_{order:02d}.json.gz')['word']
            target_word=[[g,target] for g in bare]
            control_word=[[g,control] for g in bare]
            word.extend([['cx',control,target]])
            word.extend(target_word)
            word.extend([['cx',control,target]])
            word.extend(inverse(target_word));word.extend(inverse(control_word))
        word.append(['h',target])
    return word


@lru_cache(None)
def clock(path_string):
    path=Path(path_string);row=read(path);P,m=row['P'],row['m']
    assert P==1<<m and row['width']==m
    imag=mp.iv.mpc(0,1);h=1/mp.iv.sqrt(2)
    t=mp.iv.exp(imag*mp.iv.pi/4)
    matrices={'h':mp.iv.matrix([[h,h],[h,-h]]),'t':mp.iv.matrix([[1,0],[0,t]]),
              'tdg':mp.iv.matrix([[1,0],[0,1/t]]),'s':mp.iv.matrix([[1,0],[0,imag]]),
              'sdg':mp.iv.matrix([[1,0],[0,-imag]]),'z':mp.iv.matrix([[1,0],[0,-1]]),
              'x':mp.iv.matrix([[0,1],[1,0]])}
    total=F(0);word=[['h',q] for q in range(m-1)];records=[]
    assert len(row['rotation_receipts'])==m
    for index,rec in enumerate(row['rotation_receipts']):
        multiplier=F(P-1,P) if index==0 else -F(2**(index-1),P)
        angle=mp.iv.pi*mp.iv.mpf(multiplier.numerator)/multiplier.denominator
        target=mp.iv.matrix([[mp.iv.exp(-imag*angle/2),0],[0,mp.iv.exp(imag*angle/2)]])
        U=mp.iv.eye(2)
        for gate in rec['word']:U=matrices[gate]*U
        phase=mp.iv.exp(imag*mp.iv.mpf(rec['comparison_phase']))
        difference=U-phase*target
        frobenius=mp.iv.sqrt(sum((abs(difference[i,j])**2 for i in range(2) for j in range(2)),mp.iv.mpf(0)))
        error=F(rec['interval_projective_error']);assert upper(frobenius)<=error
        assert rec['T_count']==sum(g in ('t','tdg') for g in rec['word'])
        block=[['sdg',m-1],['h',m-1]]+[[g,m-1] for g in rec['word']]+[['h',m-1],['s',m-1]]
        if index==0:
            assert rec['kind']=='fixed';word.extend(block);total+=error
        else:
            q=index-1
            assert rec['kind']=='controlled_half_angle' and rec['control']==q
            word.extend([['cx',q,m-1]]);word.extend(inverse(block));word.extend([['cx',q,m-1]]);word.extend(block)
            total+=2*error
        records.append(dict(pi_multiplier=str(multiplier),independent_frobenius_upper=str(upper(frobenius)),receipt_upper=str(error)))
    assert word==row['word'] and total==F(row['operator_error'])
    c=counts(word,m)
    assert (c['T_count'],c['gates'],c['T_depth'],c['depth'])==(row['T'],row['nonoracle_gates'],row['T_depth'],row['nonoracle_depth'])
    return dict(path=str(path),sha256=digest(path),P=P,m=m,**c,operator_error=str(total),rotation_checks=records)


def qft(path_string):
    path=Path(path_string);row=read(path);n=row['m'];word=expected_iqft(n)
    assert row['word']==word
    c=counts(word,n)
    for key in c:assert c[key]==row[key]
    eps=F(0)
    independent=read(OUT/'review_ramps.json')
    # Bind every source to the separate stdlib directed-interval receipt replay.
    receipts={r['j']:r for r in independent['raw_receipts']}
    for j in range(3,n+1):
        source=LIB/f'rotation_{j:02d}.json.gz';raw=read(source);receipt=receipts[j]
        assert digest(source)==receipt['sha256']
        assert F(receipt['direct_forward_frobenius_upper'])<=F(raw['interval_operator_upper'])
        assert F(receipt['direct_adjoint_frobenius_upper'])<=F(raw['interval_operator_upper'])
        eps+=3*(n-j+1)*F(raw['interval_operator_upper'])
    assert eps==F(row['operator_error'])
    for rec in row['rotation_sources']:
        assert digest(rec['path'])==rec['sha256']
        assert rec['multiplicity']==3*(n-rec['j']+1)
    return dict(path=str(path),sha256=digest(path),m=n,**c,operator_error=str(eps))


def review_padding(path):
    module=read(path);P,S=module['P'],module['S'];m=P.bit_length()-1;r=S.bit_length()-1;n=m+r
    assert P==1<<m and S==1<<r and module['phase_width']==n
    assert module['padded_phase_size']==P*S
    assert module['initial_high_bits']==dict(wires=list(range(m,n)),state='all zero')
    assert len(module['powers'])==m
    for k,power in enumerate(module['powers']):
        assert power['control_wire']==k and power['repeat']==2**k
    assert sum(power['repeat'] for power in module['powers'])==module['coherent_label_queries']==module['maximum_sequential_label_calls']==P-1
    cp,qp=module['clock'],module['inverse_qft']
    assert cp['local_wires']==list(range(m)) and qp['local_wires']==list(range(n))
    assert digest(cp['path'])==cp['sha256'] and digest(qp['path'])==qp['sha256']
    c=clock(cp['path']);q=qft(qp['path'])
    for reference,actual in ((cp,c),(qp,q)):
        for key in ('T_count','gates','T_depth','depth','operator_error'):assert str(reference[key])==str(actual[key])
    expected=dict(T_count=c['T_count']+q['T_count'],gates=c['gates']+q['gates'],
                  T_depth_serial_actual=c['T_depth']+q['T_depth'],depth_serial_actual=c['depth']+q['depth'],
                  T_depth_serial_gate_upper=c['T_depth']+q['T_count'],depth_serial_gate_upper=c['depth']+q['gates'])
    assert expected==module['costs']
    assert F(module['readout_operator_error'])==F(c['operator_error'])+F(q['operator_error'])
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path),P=P,S=S,phase_width=n,costs=expected,clock=c,iqft=q)


def decoder_checks():
    # This inequality proves the decoder error for every possible shifted angle,
    # independent of finite P/S sizes or the test cases below.
    truncation=(2*PI_HI)**66/F(math.factorial(66))
    perturbation=2*(PI_HI-PI_LO)
    rounding=F(1,2**49)
    assert truncation+perturbation+rounding<F(1,2**48)
    path=ACCEPTED/'src/statistic/finite_decoder.py'
    spec=importlib.util.spec_from_file_location('_review_accepted_decoder',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    wrapper_path=ROOT/'src/random_sign/native.py'
    wrapper=None
    if wrapper_path.exists():
        wrapper_spec=importlib.util.spec_from_file_location('_review_decoder_wrapper',wrapper_path)
        wrapper=importlib.util.module_from_spec(wrapper_spec);wrapper_spec.loader.exec_module(wrapper)
    cases=[]
    for P in (2,4,32,16384):
        for S in (1,2,4,16):
            for j in sorted({0,1,P//4,P//2,max(0,P-2),P-1}):
                for z in sorted({0,S//2,S-1}):
                    value=mod.signed_mean(S*j+z,S*P)
                    if wrapper is not None:assert wrapper.signed_mean(j,P,z,S)==value
                    angle=2*mp.iv.pi*mp.iv.mpf(S*j+z)/(S*P)
                    difference=abs(mp.iv.mpf(value.numerator)/value.denominator-mp.iv.cos(angle))
                    assert upper(difference)<F(1,2**48)
                    cases.append(dict(P=P,S=S,j=j,z=z,decoded=str(value),absolute_error_upper=str(upper(difference))))
    error=truncation+perturbation+rounding
    dyadic=F(-(-(error*2**80).numerator//(error*2**80).denominator),2**80)
    decisions=0
    if wrapper is not None:
        for head in (F(-2),F(-1),F(0),F(1),F(2)):
            for sample in (((0,0),),((3,3),),((0,0),(3,3),(1,2))):
                vals=sorted(mod.signed_mean(4*j+z,16) for j,z in sample)
                score=head+vals[len(vals)//2]
                assert wrapper.decision(sample,4,4,F(1),head)==(1 if score>=0 else -1)
                decisions+=1
        for args in ((0,4,-1,4),(0,4,4,4),(0,4,0,3),(-1,4,0,4),(4,4,0,4)):
            try:wrapper.signed_mean(*args)
            except ValueError:pass
            else:raise AssertionError(('invalid decoder input accepted',args))
    return dict(path=str(path),sha256=digest(path),wrapper_path=str(wrapper_path),wrapper_sha256=digest(wrapper_path) if wrapper else None,
                uniform_error_upper=str(dyadic),cases=len(cases),decision_cases=decisions,
                scope='Uniform exact Taylor/pi/rounding proof, plus boundary and sign-threshold spot checks using direct 120-digit interval cosine. Wrapper and median/tie/shift behavior separately exercised.')


def apply_native(word,vector):
    v=np.asarray(vector,dtype=complex).copy();indices=np.arange(len(v))
    invsqrt=1/math.sqrt(2)
    mats={'h':np.array([[invsqrt,invsqrt],[invsqrt,-invsqrt]]),
          'x':np.array([[0,1],[1,0]]),'z':np.diag([1,-1]),
          's':np.diag([1,1j]),'sdg':np.diag([1,-1j]),
          't':np.diag([1,np.exp(1j*math.pi/4)]),'tdg':np.diag([1,np.exp(-1j*math.pi/4)])}
    for op in word:
        if op[0]=='cx':
            _,control,target=op
            left=indices[((indices>>control)&1).astype(bool)&~((indices>>target)&1).astype(bool)]
            right=left^(1<<target);v[left],v[right]=v[right].copy(),v[left].copy()
        else:
            name,wire=op;left=indices[(indices&(1<<wire))==0];right=left|(1<<wire)
            a,b=v[left].copy(),v[right].copy();M=mats[name]
            v[left]=M[0,0]*a+M[0,1]*b;v[right]=M[1,0]*a+M[1,1]*b
    return v


def padding_semantics():
    module=read(OUT/'blind_padding_module_P32_S4.json');P,S=32,4;Q=P*S
    cw=read(module['clock']['path'])['word'];qw=read(module['inverse_qft']['path'])['word']
    zero=np.zeros(Q,complex);zero[0]=1
    native=apply_native(cw,zero);assert np.all(native[P:]==0)
    ideal=np.zeros(Q,complex);ideal[:P]=np.sqrt(2/P)*np.sin(math.pi*(np.arange(P)+.5)/P)
    bound=float(F(module['readout_operator_error']));worst=0.;equivalence=0.;cases=0
    for phi in (0.,math.pi/P,.317,math.pi/2-.001,math.pi/2+.001,math.pi-.001,2*math.pi-.001):
        for theta in (phi,-phi):
            phase=np.ones(Q,complex);phase[:P]=np.exp(1j*np.arange(P)*theta)
            actual=apply_native(qw,native*phase)
            expected=np.fft.fft(ideal*phase)/math.sqrt(Q)
            phase_align=np.vdot(expected,actual)
            distance=float(np.linalg.norm(actual*np.exp(-1j*np.angle(phase_align))-expected))
            assert distance<=bound+1e-11;worst=max(worst,distance)
            shifted=np.empty(Q)
            for z in range(S):
                state=ideal[:P]*np.exp(1j*np.arange(P)*(theta-2*math.pi*z/Q))
                shifted[z::S]=abs(np.fft.fft(state)/math.sqrt(P))**2/S
            discrepancy=float(np.max(abs(shifted-abs(expected)**2)))
            assert discrepancy<1e-12;equivalence=max(equivalence,discrepancy);cases+=1
    return dict(cases=cases,maximum_native_state_error=worst,certified_readout_operator_error=module['readout_operator_error'],
                maximum_randomized_padded_probability_difference=equivalence,scope='Independent simulator of actual emitted words at P32/S4; numerical falsification only, including both eigenbranches and half-bin points.')


def main():
    paths=sorted(OUT.glob('blind_padding_module_P*_S*.json'));assert paths
    rows=[review_padding(p) for p in paths]
    result=dict(status='passed',padding_modules=len(rows),rows=rows,decoder=decoder_checks(),native_semantics=padding_semantics(),
                scope='Literal padding clock/IQFT reconstruction and receipt errors; low-only controlled-power schedules; declared readout costs. Complete root program ledgers are a separate check.')
    (OUT/'review_native_programs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows',)}))


if __name__=='__main__':main()
