"""Read-only accepted dependencies and new coordinate artifact helpers."""
from pathlib import Path
from fractions import Fraction
import sys,json,gzip
ROOT=Path(__file__).resolve().parents[2]
ACCEPTED=ROOT.parent/'heat_weight_response_frontier'
for path in [ROOT/'src',ROOT/'src/response',ROOT/'src/statistic',ROOT.parent/'heat_weight_shared_observable/src',ROOT.parent/'heat_weight_unified_frontier/src']:
    sys.path.insert(0,str(path))
import coherent_compile as cc
import autonomous_blind_source as source
import experiment as ex
from shared_observable.boolean_circuit import Circuit,ledger,inverse,sparse_apply,classical_apply
OUT=ROOT/'results/coordinate'
OUT.mkdir(parents=True,exist_ok=True)
def load(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_text(encoding='utf-8-sig'))
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(obj,indent=None if path.suffix=='.gz' else 2,default=str)+'\n'
    if path.suffix=='.gz':
        with gzip.open(path,'wt',encoding='utf8') as f:f.write(text)
    else:path.write_text(text,encoding='utf8')
def saved(tag,method):
    comparison=load(ACCEPTED/'results/response'/f'{tag}_final_comparison.json')
    row=next(x for x in comparison['selected'] if x['method']==method and x['equal_source'])
    name=row['artifact']
    return row,load(ACCEPTED/'results/response'/(name+'_raw.json.gz')),load(ACCEPTED/'results/response'/(name+'_word.json.gz'))
def thaw(raw):
    qc=Circuit();qc.width=raw['width'];qc.regs={k:list(v) for k,v in raw['registers'].items()}
    qc.scratch=[q for k,v in qc.regs.items() if k.startswith('scratch_') for q in v]
    B=[tuple(o) for o in raw['B']];phase=raw['controlled_phase'][len(B):-len(B)]
    return dict(qc=qc,B=B,phase=[tuple(o) for o in phase],seeds=list(raw['seedbits']),
      outer=raw['outer_control'],target=raw['phase_target'],
      source_preparation=[tuple(o) for o in raw['source_preparation']],source_receipt=raw['source_receipt'],
      source_ordinary_bits=raw['source_ordinary_bits'],source_shell_bits=raw['source_shell_bits'])
def boundaries(raw):
    """Locate prefix segments by actual emitted word and documented register uses."""
    B=[tuple(o) for o in raw['B']];r=raw['registers'];n=len(r['coordinate']);shells=raw['descriptor']['q_tail']['shell_exponents']
    start=raw['source_selector_receipt']['new_macro_gates']
    assert all(o[0] in ('x','cx','ccx') for o in B)
    jword=[]
    for k in shells:
        en=r['shell_'+str(k)][0];jword.append(('cx',en,r['coordinate'][k]))
        jword += [('ccx',en,r['seed_offset'][i],r['coordinate'][i]) for i in range(k)]
    assert B[start:start+len(jword)]==jword
    jend=start+len(jword)
    alpha=set(r['side_fraction']);frac=set(r['normalized_fraction'])
    astart=next(i for i in range(jend,len(B)) if B[i][-1] in alpha)
    fstart=next(i for i in range(astart,len(B)) if B[i][-1] in frac)
    fend=fstart
    while fend<len(B) and B[fend][0]=='ccx' and B[fend][-1] in frac:fend+=1
    threshold=set(r['accept_threshold'])|set(r.get('unit_probability_threshold',[]))|set(r['negative_threshold'])
    # QROM begins before its first threshold write; use producer's complete
    # accepted chunk lengths, checking exhaustive partition against B.
    return dict(selector=(0,start),offset_mask=(start,jend),signed_address=(jend,astart),
                side_fraction=(astart,fstart),probability_window=(fstart,fend),
                suffix_start=fend)
