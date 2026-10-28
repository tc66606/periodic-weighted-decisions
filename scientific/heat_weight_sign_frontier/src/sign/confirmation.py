"""Two genuinely new fixed public tuples; inherited observable, new readout only."""
from pathlib import Path
import sys,json,gzip,time,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src/coordinate'))
from common import ex,source,thaw,Fraction
from fresh import make_raw
from joint import transform
OUT=ROOT/'results/sign/confirmation'
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(obj,default=str,indent=None if p.suffix=='.gz' else 2)+'\n').encode()
    p.write_bytes(gzip.compress(data,mtime=0) if p.suffix=='.gz' else data)
def run():
    reg=json.loads((ROOT/'configs/sign/confirmation.json').read_text())
    for name,sha in reg['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'Locked readout changed'
    rows=[]
    for j,point in enumerate(reg['points']):
        start=time.perf_counter();n=point['n'];data=ex.ml.fixed_data(1<<n,point['parameters']);R=data['radius']
        d,head,pre=ex.api.bc.tail_description(data,R);eb=Fraction(d['eta']).denominator.bit_length()-1
        f=min(22,max(d['q_tail']['shell_exponents'])+eb);attempts=[]
        for method in ('linear','nodes'):
            best=None
            for g in range(1,min(8,f)+1):
                precision=dict(c=28,g=g,p=23,round_bits=f-g,merge_small_cut=10)
                if method=='nodes':precision['endpoint_layout']='shared'
                b=ex.build(d,method,precision)
                if method=='nodes':ex.cell_refinement(b,d)
                b=source.transform(b,d);bias=data['approximation_bias']+b['bias']
                if bias>=Fraction('.0002'):
                    attempts.append(dict(method=method,g=g,bias=str(bias),feasible=False));continue
                raw=make_raw(b,d,head,pre,data['approximation_bias']);raw=transform(raw,'parallel' if f%2==0 else 'tapered')
                comp=source.compile_source(thaw(raw));sch=ex.choose(b['Z'],bias,comp['compiled_ledger']['T_count'])
                total=ex.complete_totals(comp,sch['P'],sch['K'],2*R)
                row=dict(tag=f'heldout_{j}',method=method,n=n,parameters=point['parameters'],g=g,precision=precision,
                    bias=str(bias),Z=str(b['Z']),totals=total,schedule=sch,head_preprocessing=pre)
                attempts.append(dict(method=method,g=g,bias=str(bias),feasible=True,T_count=total['T_count']))
                if best is None or total['T_count']<best[0]['totals']['T_count']:best=(row,raw,comp)
            assert best
            row,raw,comp=best;name=f'heldout_{j}_{method}';row['artifact']=name
            save(OUT/(name+'_raw.json.gz'),raw);save(OUT/(name+'_word.json.gz'),comp);save(OUT/(name+'.json'),row)
            rows.append(row);print(name,row['totals']['T_count'],flush=True)
        save(OUT/f'heldout_{j}_attempts.json',dict(point=point,attempts=attempts,seconds=time.perf_counter()-start))
    save(OUT/'index.json',dict(rows=rows,registration=reg))
if __name__=='__main__':run()
