"""Exact joint shell-address and probability-coordinate transformation."""
from common import *
from seeded_layout import emit_mask_coordinate,emit_signed_address
from blind_window import tapered_window,funnel_window
import copy,hashlib,time

def transform(raw,router='tapered'):
    old=copy.deepcopy(raw);B=[tuple(o) for o in old['B']];r=old['registers'];bounds=boundaries(old)
    d=old['descriptor'];ks=d['q_tail']['shell_exponents']
    assert not d['head'] and ks==list(range(min(ks),max(ks)+1))
    qc=thaw(old)['qc'];qc.ops=B[:bounds['selector'][1]]
    chunks={'selector':(0,len(qc.ops))}
    def mark(name,start):chunks[name]=(start,len(qc.ops))
    start=len(qc.ops)
    mask=emit_mask_coordinate(qc,{k:r['shell_'+str(k)][0] for k in ks},r['seed_offset'],r['coordinate'])
    mark('offset_mask',start);start=len(qc.ops)
    signed=emit_signed_address(qc,r['coordinate'],r['tail'][0],r['positive_side'][0],r['address'],r['public_sign'][0])
    mark('signed_address',start);start=len(qc.ops)
    oldneg=r['negative_side'][0];newneg=signed['negative']
    negmap={i:(newneg if i==oldneg else i) for i in range(qc.width)}
    qc.ops+=cc.remap_word(B[slice(*bounds['side_fraction'])],negmap)
    r['negative_side']=[newneg];mark('side_fraction',start);start=len(qc.ops)
    oldfrac=list(r['normalized_fraction'])
    if router=='tapered':
        frac,route=tapered_window(qc,r['seed_offset'],r['side_fraction'],r['equal_shell_source'],len(oldfrac),min(ks),oldfrac)
    elif router.startswith('funnel'):
        frac,route=funnel_window(qc,r['seed_offset'],r['side_fraction'],r['equal_shell_source'],len(oldfrac),min(ks),router=='funnel_clean')
    elif router=='parallel':
        frac,route=tapered_window(qc,r['seed_offset'],r['side_fraction'],r['equal_shell_source'],len(oldfrac),min(ks),oldfrac,parallel=True,scratch_pool=qc.scratch)
    else:raise ValueError(router)
    mark('probability_window',start)
    remapping={q:q for q in range(qc.width)};remapping.update(zip(oldfrac,frac))
    later=B[bounds['suffix_start']:]
    old_inputs=set(r['seed_offset']+r['side_fraction']+r['equal_shell_source'])
    assert not any(old_inputs.intersection(o[1:]) for o in later)
    qc.ops+=cc.remap_word(later,remapping);mark('cell_lookup_and_coins',chunks['probability_window'][1])
    phase=[tuple(o) for o in old['controlled_phase'][len(B):-len(B)]]
    newB=qc.ops.copy();r['normalized_fraction']=list(frac)
    old.update(B=newB,controlled_phase=newB+phase+newB[::-1],width=qc.width,registers=r)
    old['coordinate_construction']=dict(router=router,mask=mask,signed=signed,window=route,chunk_ranges=chunks,
      original_B_sha256=hashlib.sha256(json.dumps(raw['B'],separators=(',',':')).encode()).hexdigest(),
      exact_scope='For every valid equal-shell source code and every ordinary seed, address, sign, fraction, threshold and acceptance match. Offset/alpha permutation retained to literal B inverse before source reflection.',
      unchanged_seedbits=old['seedbits']==raw['seedbits'],source_approximation_scope='Same ideal bounded observable on valid source support; unchanged 2Z epsilon bound covers invalid codes of the same fixed synthesized source.')
    return old

def run_saved(tag,method,router='tapered',persist=True):
    start=time.perf_counter();row,old,prior=saved(tag,method)
    new=transform(old,router);comp=source.compile_source(thaw(new))
    Z=Fraction(new['normalization_fraction']);bias=Fraction(new['raw_bias_fraction'])
    sched=ex.choose(Z,bias,comp['compiled_ledger']['T_count'])
    assert sched
    total=ex.complete_totals(comp,sched['P'],sched['K'],row['totals']['measured_head_queries'])
    sched['T']=total['T_count']
    name=tag+'_'+method+'_'+router
    result=dict(name=name,tag=tag,method=method,router=router,source_artifact=row['artifact'],
      Z=str(Z),bias=str(bias),raw_error=sched['raw_error'],failure=sched['failure'],schedule=sched,
      totals=total,control_totals=row['totals'],construction=new['coordinate_construction'],
      seconds=time.perf_counter()-start,scope='Complete same-head same-label-oracle promised sign. Native per-iterate word emitted; finite repetitions counted, not quantum hardware execution.')
    if persist:
        save(OUT/(name+'_raw.json.gz'),new);save(OUT/(name+'_word.json.gz'),comp);save(OUT/(name+'.json'),result)
    return result,new,comp

if __name__=='__main__':
    rows=[]
    for tag in ['discovery_n12','discovery_n68','confirmation_0','confirmation_1']:
        for method in ['linear','nodes']:
            result,_,_=run_saved(tag,method)
            rows.append(result)
            print(json.dumps({k:result[k] for k in ['name','seconds']}|{'T':result['totals']['T_count'],'Tdepth':result['totals']['T_depth_upper'],'depth':result['totals']['nonoracle_depth_upper'],'width':result['totals']['allocated_width'],'queries':result['totals']['label_queries']}),flush=True)
    save(OUT/'joint_initial_comparison.json',dict(rows=rows))
