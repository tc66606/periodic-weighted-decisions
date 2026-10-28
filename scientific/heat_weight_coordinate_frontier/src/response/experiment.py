"""Same-task response comparisons: one descriptor, own bias, full phase schedule."""
from pathlib import Path
from fractions import Fraction as F
import gzip,json,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/statistic'))
sys.path.insert(0,str(Path(__file__).parent))
import blind_compiled as api
import comparison_table as ct
import hermite_sampler as hs
from autonomous_blind_nodes import nodes_provider
from response.hermite_higher import quintic_tables,septic_tables
bi=api.bi;ml=api.ml
OUT=ROOT/'results/response'

def dump(path,x):
    path.parent.mkdir(exist_ok=True,parents=True)
    text=json.dumps(x,indent=None if path.suffix=='.gz' else 2,default=str)+'\n'
    if path.suffix=='.gz':
        with gzip.open(path,'wt',encoding='utf8') as f:f.write(text)
    else:path.write_text(text,encoding='utf8')

def choose(Z,bias,T):
    if bias>=ml.RAW_BIAS_MAX:return None
    choices=[]
    for m in range(1,19):
        P=1<<m;qft=ml.qft_cost(m)
        err=bias+ml.finite_error(Z,P)+Z*F(1,1<<48)
        if err>=ml.RAW_MARGIN:continue
        for K in range(1,24,2):
            failure=ml.median_failure(K)+2*K*qft['operator_error']+ml.LOGICAL_RESERVE
            if failure<ml.FAIL_MAX:choices.append({'P':P,'K':K,'raw_error':str(err),'failure':str(failure),'queries':K*(P-1),'T':K*((P-1)*T+qft['T_count'])})
    return min(choices,key=lambda x:x['T']) if choices else None

def build(d,kind,precision):
    if kind=='linear':return bi.interp_sampler(d,centered=True,retain_boundaries=True,**precision)
    return hs.hermite_sampler(d,retain_boundaries=True,table_provider={'nodes':nodes_provider,'quintic':quintic_tables,'septic':septic_tables}.get(kind),method=kind,**precision)

def cell_refinement(b,d):
    """Average local absolute remainder over exactly occupied uniform cells.

    This is an all-label raw bias bound, not a smaller pointwise error.
    Separate source approximation is still added after this ideal-proposal bound.
    """
    degree=b['config']['degree'];g=b['config']['g'];M=1<<g
    factor=sum((F(M,M+j)**(degree+2) for j in range(M)),F(0))/M
    mass=F(d['q_tail']['per_side_shell_mass_numerator'],int(d['dyadic_denominator']))
    new=2*mass*b['interp_error']*sum((factor if k>=g else F(1) for k in b['shells']),F(0))
    old=b['bias_components']['hermite'];b['bias']+=new-old;b['bias_components']['hermite']=new
    b['cell_refinement']={'uniform_cell_factor':str(factor),'old_raw_remainder':str(old),'new_raw_remainder':str(new),'derivative_order':degree+1,'hypotheses':'0<eta<1; shell k>=g has exactly2^(k-g) offsets/cell; k<g uses global fallback. Coordinate flooring preserves cell. Uniform pointwise bound remains unchanged.'}
    return b

def complete_totals(comp,P,K,head):
    total=ct.totals([(comp,P,K)],head)
    if 'initial_preparation' in comp:
        init=comp['initial_preparation'];from collections import Counter
        cnt=Counter(x[0] for x in init);d,t=ct.literal_depth(init,comp['width'])
        total['T_count']+=K*(cnt['t']+cnt['tdg']);total['nonoracle_gates']+=K*(len(init)-len(comp['seedbits']))
        total['T_depth_upper']+=K*t;total['nonoracle_depth_upper']+=K*(d-1)
        total['initial_source_per_run']={'T':cnt['t']+cnt['tdg'],'T_depth':t,'nonoracle_depth':d,'gates':len(init)}
    return total

def run_one(n,R,params,kind,precision,tag,verify=False,persist=True,data=None):
    start=time.perf_counter();data=data or ml.fixed_data(1<<n,params)
    d,head,pre=api.bc.tail_description(data,R);b=build(d,kind,precision)
    raw=hs.bridge(b);raw.update(descriptor=d,head_coefficients=head,head_preprocessing=pre,configuration=b['config'],endpoint_table=b['table'],table_keys=b['table_keys'],endpoint_intervals=b['table_intervals'],normalization_fraction=str(b['Z']))
    comp=api.compile_bundle(raw);bias=data['approximation_bias']+b['bias'];raw['raw_bias_fraction']=str(bias)
    raw['fixed_coefficient_and_tail_bias']=str(data['approximation_bias'])
    if 'bias_components' in b:raw['bias_components']={k:str(v) for k,v in b['bias_components'].items()}
    schedule=choose(b['Z'],bias,comp['compiled_ledger']['T_count'])
    row={'tag':tag,'n':n,'R':R,'parameters':params,'method':kind,'precision':precision,'bias':str(bias),'bias_float':float(bias),'Z':str(b['Z']),'schedule':schedule,'chunks':b['chunks'],'head_preprocessing':pre,'elapsed_seconds':time.perf_counter()-start}
    if schedule:row['totals']=ct.totals([(comp,schedule['P'],schedule['K'])],2*R)
    row['iterate_T']=comp['compiled_ledger']['T_count'];row['source_seeds']=len(comp['seedbits']);row['table_entries']=sum(len(v)*(1 if kind=='linear' else b['config']['degree']+1) for v in b['table'].values())
    if verify:
        row['verification']=bi.check(b,d,trials=30) if kind=='linear' else hs.check(b,d,trials=30,compiled=comp)
        if n<=13:row['numerical_law']=bi.numerical_law(b,d) if kind=='linear' else hs.numerical_law(b,d)
    if persist:
        dump(OUT/(tag+'.json'),row);dump(OUT/(tag+'_raw.json.gz'),raw);dump(OUT/(tag+'_word.json.gz'),comp)
    return row

def register():
    seed='heat-weight-response-frontier/fresh-confirmation/2026-09-28';digest=hashlib.sha256(seed.encode()).digest()
    choices={'tau':['.875','1.125','1.375'],'lam':['.075','.125','.175'],'eta':['.28125','.34375','.40625']}
    points=[]
    for j,n in enumerate([11,13]):
        points.append({'n':n,'parameters':{k:v[digest[3*j+i]%len(v)] for i,(k,v) in enumerate(choices.items())}})
    obj={'fixed_string':seed,'sha256':digest.hex(),'points':points,'rules':'No labels or observed margins used. Generate the tuple-specific certified heat descriptor. Fixed raw margin .002/bias .0002/failure .05. Source/response precision selected by exact public error and complete T over declared menu; all attempts including infeasible retained. Old N8192 frozen-menu result is not overwritten.'}
    path=ROOT/'configs/response/confirmation.json'
    if path.exists():assert json.loads(path.read_text())==obj
    else:dump(path,obj)
    return obj

def initial():
    register();rows=[]
    for n,R in [(12,128),(68,2048)]:
        params={'tau':'1','lam':'.1','eta':'.375'};data=ml.fixed_data(1<<n,params)
        for kind in ('hermite','nodes'):
            for g in ([1,2,3] if n==12 else [3,4,5]):
                precision=dict(c=22 if n==12 else 28,g=g,p=18 if n==12 else 23,round_bits=None if n==12 else 18,merge_small_cut=None if n==12 else 10)
                tag=f'initial_n{n}_{kind}_g{g}';row=run_one(n,R,params,kind,precision,tag,data=data,persist=False)
                rows.append(row);print(json.dumps({k:row[k] for k in ['tag','bias_float','iterate_T','schedule']}),flush=True)
        feasible=[x for x in rows if x['n']==n and x['schedule']]
        best=min(feasible,key=lambda x:x['schedule']['T']);run_one(n,R,params,best['method'],best['precision'],f'initial_best_n{n}',data=data,verify=True)
    dump(OUT/'initial_comparisons.json',rows)

if __name__=='__main__':
    if '--register' in sys.argv:print(json.dumps(register()))
    else:initial()
