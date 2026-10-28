"""Finite exact-rational classical postprocessing for a measured QPE outcome.

The Taylor truncation and PI_UP argument perturbation are interval-certified
independently. Nearest48-bit rounding leaves total mean error below2**-48.
This is classical work, not part of the quantum Clifford+T ledger.
"""
from fractions import Fraction
PI_UP=Fraction('3.14159265358979323846264338327950288419716939937511')

def signed_mean(outcome,phase_size,bits=48):
    if phase_size<2 or phase_size&(phase_size-1) or not 0<=outcome<phase_size:
        raise ValueError('Require a power-of-two phase size and its measured outcome.')
    if bits!=48:raise ValueError('Only the certified48-bit decoder is exposed.')
    x=2*PI_UP*outcome/phase_size;term=Fraction(1);value=term
    for k in range(1,33):
        term*= -x*x/Fraction((2*k-1)*(2*k));value+=term
    scaled=value*(1<<bits)
    integer=(2*scaled.numerator+scaled.denominator)//(2*scaled.denominator)
    return Fraction(integer,1<<bits)

def decode_score(outcomes,phase_size,normalization,known_head=Fraction(0)):
    if not outcomes or len(outcomes)%2!=1:raise ValueError('Require an odd nonempty repetition list.')
    values=sorted(signed_mean(j,phase_size) for j in outcomes)
    return Fraction(known_head)+Fraction(normalization)*values[len(values)//2]

if __name__=='__main__':
    import json
    from pathlib import Path
    rows=[]
    for P in (2,4,8,16,512,2048,8192,16384,32768):
        for j in sorted({0,1,P-1,P//2,P//4,3*P//4,max(0,P//4-1),min(P-1,P//4+1)}):
            rows.append({'P':P,'j':j,'signed_mean_fraction':str(signed_mean(j,P))})
    path=Path(__file__).resolve().parents[2]/'results/statistic/finite_decoder_witness.json'
    path.write_text(json.dumps({'algorithm':'Exact rational cosine Taylor through degree64, nearest48bit dyadic; classical computation',
                                'uniform_mean_error_claim':'1/281474976710656','cases':rows},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'witnesses':len(rows),'path':str(path)}))
