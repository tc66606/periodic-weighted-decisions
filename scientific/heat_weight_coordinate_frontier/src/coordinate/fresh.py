"""Two predeclared fresh public parameter points; finite precision selection."""
from common import *
from joint import transform
import time,math

def make_raw(b,d,head,pre,fixed):
    raw=ex.hs.bridge(b)
    raw.update(descriptor=d,head_coefficients=head,head_preprocessing=pre,
      configuration=b['config'],endpoint_table=b['table'],table_keys=b['table_keys'],
      endpoint_intervals=b['table_intervals'],normalization_fraction=str(b['Z']),
      raw_bias_fraction=str(fixed+b['bias']),fixed_coefficient_and_tail_bias=str(fixed),
      cell_refinement=b.get('cell_refinement'),source_preparation=b['source_preparation'],
      source_receipt=b['source_receipt'],source_selector_receipt=b['source_selector_receipt'],
      source_ordinary_bits=b['source_ordinary_bits'],source_shell_bits=b['source_shell_bits'])
    if 'bias_components' in b:raw['bias_components']={k:str(v) for k,v in b['bias_components'].items()}
    return raw

def one(point,index):
    t0=time.perf_counter();n=point['n'];data=ex.ml.fixed_data(1<<n,point['parameters'])
    R=data['radius'];d,head,pre=ex.api.bc.tail_description(data,R)
    eb=Fraction(d['eta']).denominator.bit_length()-1
    f=min(22,max(d['q_tail']['shell_exponents'])+eb)
    attempts=[];chosen={}
    for method in ('linear','nodes'):
        for g in range(1,min(8,f)+1):
            precision=dict(c=28,g=g,p=23,round_bits=f-g,merge_small_cut=10)
            if method=='nodes':precision['endpoint_layout']='shared'
            b=ex.build(d,method,precision)
            if method=='nodes':ex.cell_refinement(b,d)
            b=source.transform(b,d);bias=data['approximation_bias']+b['bias']
            if bias>=ex.ml.RAW_BIAS_MAX:
                attempts.append(dict(method=method,g=g,bias=str(bias),feasible=False,reason='raw bias cap'));continue
            raw=make_raw(b,d,head,pre,data['approximation_bias'])
            comp=source.compile_source(thaw(raw))
            oldschedule=ex.choose(b['Z'],bias,comp['compiled_ledger']['T_count'])
            assert oldschedule
            oldtotal=ex.complete_totals(comp,oldschedule['P'],oldschedule['K'],2*R)
            mode='parallel' if f%2==0 else 'tapered'
            new=transform(raw,mode);compiled=source.compile_source(thaw(new))
            schedule=ex.choose(b['Z'],bias,compiled['compiled_ledger']['T_count'])
            total=ex.complete_totals(compiled,schedule['P'],schedule['K'],2*R);schedule['T']=total['T_count']
            row=dict(method=method,g=g,precision=precision,router=mode,bias=str(bias),Z=str(b['Z']),schedule=schedule,
                     totals=total,prior_totals=oldtotal,prior_schedule=oldschedule,feasible=True)
            attempts.append(row)
            # Keep the best old and new method separately; no mixed certificate.
            for variant,tot,artifact in [('new',total,(new,compiled)),('prior',oldtotal,(raw,comp))]:
                key=method+'_'+variant
                if key not in chosen or tot['T_count']<chosen[key][0]['selected_total']['T_count']:
                    pick=dict(row,variant=variant,selected_total=tot)
                    chosen[key]=(pick,*artifact)
        print('Fresh',index,method,'compiled',flush=True)
    selected=[]
    for key,(row,raw,compiled) in chosen.items():
        name=f'fresh_{index}_{key}';row['artifact']=name;selected.append(row)
        save(OUT/(name+'_raw.json.gz'),raw);save(OUT/(name+'_word.json.gz'),compiled)
        save(OUT/(name+'.json'),row)
    report=dict(index=index,n=n,parameters=point['parameters'],R=R,physical_fraction_bits=f,
                seconds=time.perf_counter()-t0,attempts=attempts,selected=selected,head_preprocessing=pre,
                scope='Two fixed public tuples; actual interval descriptors and native words. Prior and new each choose own feasible complete schedule; no unknown labels used.')
    save(OUT/f'fresh_{index}_comparison.json',report)
    print(json.dumps({'index':index,'seconds':report['seconds'],'selected':[dict(method=r['method'],variant=r['variant'],g=r['g'],T=r['selected_total']['T_count'],width=r['selected_total']['allocated_width'],queries=r['selected_total']['label_queries']) for r in selected]}),flush=True)
    return report

if __name__=='__main__':
    config=load(ROOT/'configs/coordinate/confirmation.json')
    for j,p in enumerate(config['points']):one(p,j)
