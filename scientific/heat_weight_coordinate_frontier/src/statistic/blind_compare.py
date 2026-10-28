"""Matched measured-head controls and fixed-certificate radius scan.

Every method receives the same measured head, shifted decision threshold,
coefficient approximation, omitted smooth tail certificate, and XOR oracle.
"""
from fractions import Fraction
from pathlib import Path
import copy, gzip, json, math, sys, time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import blind_interp as bi
ml=bi.ml
import precision_sampler as prefix
from shared_observable.comparator_sampler import build as old_emit
from shared_observable.blind_practical_descriptor import round_dyadic, upper_text

def tail_description(data,R):
    base_radius=data['radius']
    if R<base_radius:raise ValueError('Requested head radius is below this tuple\'s certified original descriptor radius.')
    d=ml.level_descriptor(data,0,base_radius);head=d['head'];d['head']=[]
    started=time.perf_counter();extra=Fraction(0);mp=bi.mp;iv=mp.iv
    eta=iv.mpf(d['eta']);N=d['N'];sin=iv.sin(iv.pi*eta);hp=1/(1+iv.mpf(d['lambda'])*iv.exp(4*iv.mpf(d['tau'])))
    for j in range(base_radius,R):
        for r in (-j,j+1):
            q=(-1 if r%2==0 else 1)*sin/(N*iv.tan(iv.pi*(r-eta)/N))
            value,error=round_dyadic(hp*q,48);extra+=Fraction(upper_text(error))
            head.append({'address':r%N,'signed_numerator':value})
    d['q_tail']['shell_exponents']=list(range(R.bit_length()-1,d['n']-1))
    d['common_head_radius']=R
    J=len(d['q_tail']['shell_exponents']);mass=d['q_tail']['per_side_shell_mass_numerator']
    d['normalization_numerator']=str(2*J*mass);d['normalization']=str(Fraction(2*J*mass,int(d['dyadic_denominator'])))
    d['descriptor_entries']=2*J
    d['error']['all_label_raw_bias_upper']=str(Fraction(d['error']['all_label_raw_bias_upper'])+extra)
    return d,head,{'head_entries':len(head),'head_label_queries':len(head),'head_numerical_weight_storage_bits':len(head)*49,
        'head_explicit_address_and_weight_bits':len(head)*(d['n']+49),'classical_accumulator_bits_sufficient':49+math.ceil(math.log2(max(1,len(head)))),
        'original_descriptor_radius':base_radius,'coefficient_degree':data['base']['coefficient_degree'],
        'extra_cardinal_weight_evaluations':2*(R-base_radius),'extra_weight_generation_seconds':time.perf_counter()-started,
        'extra_head_rounding_error_upper':str(extra),'head_weight_label_products':len(head),'head_scalar_additions':len(head)-1,
        'coefficient_generation':'Actual tuple-specific interval descriptor generated once, then bq-only public weights beyond its original radius. Its original omitted smooth residual stays charged for every R.'}

def choose_schedule(Z,bias,T):
    options=[]
    for m in range(1,18):
        P=1<<m;err=bias+ml.finite_error(Z,P)
        if err>=ml.RAW_MARGIN:continue
        qft=ml.qft_cost(m)
        for k in range(1,18,2):
            fail=ml.median_failure(k)+2*k*qft['operator_error']+ml.LOGICAL_RESERVE
            if fail>=ml.FAIL_MAX:continue
            options.append({'P':P,'k':k,'T7':k*((P-1)*T+qft['T_count']),'quantum_queries':k*(P-1),'raw_error':float(err),'failure':float(fail)})
    return min(options,key=lambda v:v['T7'])

def old_method(d,fixed,kind,compiler=None):
    # Finite control menu chosen only from raw-bias feasibility and emitted costs.
    # Original probability prefixes remain legal; no address truncation occurs.
    best=None
    for c in (24,25,26,27,28):
        for C,D in ((16,8),(18,9),(20,10)):
            if kind=='full':
                cfg=bi.configuration(d,selection_bits=c,cut=C,degree=D)
            else:
                try:cfg=prefix.configuration(d,16 if d['n']>12 else 13,14 if d['n']>12 else 13,raw_budget=str(ml.RAW_BIAS_MAX-fixed),selection_bits=c,cut=C,degree=D)
                except ValueError:continue
            bias=fixed+Fraction(cfg['raw_bias_upper_fraction'])
            if bias>=ml.RAW_BIAS_MAX:continue
            for retain in (False,True):
                if retain:
                    from control_retained import retained_build
                    module=sys.modules[old_emit.__module__] if kind=='full' else prefix
                    bundle=retained_build(module,d,cfg)
                else:bundle=old_emit(d,cfg) if kind=='full' else prefix.build(d,cfg)
                # Exact T proxy for the common compiler: B/inverse use4T per
                # CCX each; six exact phase CCX cost42; KG uses8S-1T. Every
                # feasible precision/retention menu member gets a P/k search.
                cnt=bi.counts(bundle['controlled_iterate']) if compiler is None else {'T7':8*bi.counts(bundle['B'])['ccx']+8*len(bundle['seedbits'])+41}
                Z=Fraction(int(d['normalization_numerator']),int(d['dyadic_denominator']))
                schedule=choose_schedule(Z,bias,cnt['T7'])
                config=dict(cfg,retain_selector=retain)
                row={'kind':kind,'cfg':config,'bias':bias,'counts':cnt,'schedule':schedule,'bundle':bundle,'compiled':None}
                if best is None or schedule['T7']<best['schedule']['T7']:best=row
    if compiler is not None:
        best['compiled']=compiler(best['bundle'])
        assert best['compiled']['compiled_ledger']['T_count']==best['counts']['T7']
    return best

def run():
    reports=[]
    for n in (12,68):
        data=ml.fixed_data(1<<n,{'tau':'1','lam':'.1','eta':'.375'});fixed=data['approximation_bias']
        for R in ((128,) if n==12 else (128,512,1024,2048)):
            d,head,pre=tail_description(data,R);Z=Fraction(int(d['normalization_numerator']),int(d['dyadic_denominator']))
            new=bi.interp_sampler(d,c=22 if n==12 else 25,g=5 if n==12 else 7,p=18 if n==12 else 21,centered=True,round_bits=None if n==12 else 12,retain_boundaries=True)
            bias=fixed+new['bias'];cnt=bi.counts(new['iterate']);sched=choose_schedule(Z,bias,cnt['T7'])
            methods=[{'kind':'stochastic_interpolation','configuration':new['config'],'bias':float(bias),'counts':cnt,'schedule':sched,
                'seed_bits':len(new['seeds']),'width_with_phase':new['qc'].width+sched['P'].bit_length()-1,'B_CCX':bi.counts(new['B'])['ccx'],
                'phase_CCX':bi.counts(new['phase'])['ccx'],'reflection_CCX':bi.counts(new['reflect'])['ccx']}]
            tag=f'n{n}_R{R}'
            for kind in ('full','prefix'):
                old=old_method(d,fixed,kind);bundle=old['bundle'];blen=len(bundle['B']);phase=bundle['controlled_phase'][blen:-blen]
                methods.append({'kind':kind,'configuration':old['cfg'],'bias':float(old['bias']),'counts':old['counts'],'schedule':old['schedule'],
                    'seed_bits':len(bundle['seedbits']),'width_with_phase':bundle['circuit'].width+old['schedule']['P'].bit_length()-1,
                    'B_CCX':bi.counts(bundle['B'])['ccx'],'phase_CCX':bi.counts(phase)['ccx'],'reflection_CCX':bi.counts(bundle['controlled_reflection'])['ccx']})
                with gzip.open(bi.OUT/f'blind_compare_{tag}_{kind}_word.json.gz','wt',encoding='utf-8') as stream:json.dump({'B':bundle['B'],'phase':phase,'reflection':bundle['controlled_reflection'],'registers':bundle['circuit'].regs,'seeds':bundle['seedbits'],'config':old['cfg']},stream)
            for method in methods:method['total_label_queries']=2*R+method['schedule']['quantum_queries']
            result={'n':n,'R':R,'normalization':str(Z),'fixed_coefficient_and_original_R128_tail_bias':str(fixed),'head_preprocessing':pre,
                'methods':methods,'new_verification':bi.check(new,d,trials=100),'scope':'Matched measured head and shifted threshold; total classical head queries charged. Full/prefix control menu c24..28, (C,D)=(16,8),(18,9),(20,10), same public data. Exact N-query classical solution remains stronger atN4096.'}
            bi.put(bi.OUT/f'blind_compare_{tag}.json',result);reports.append(result)
            with gzip.open(bi.OUT/f'blind_compare_{tag}_new_word.json.gz','wt',encoding='utf-8') as stream:json.dump({'B':new['B'],'phase':new['phase'],'reflection':new['reflect'],'registers':new['qc'].regs,'seeds':new['seeds'],'table':new['table'],'config':new['config'],'head_coefficients':head},stream)
            print(json.dumps({'n':n,'R':R,'head_queries':2*R,'methods':[{k:v for k,v in m.items() if k in ('kind','schedule','total_label_queries','width_with_phase','B_CCX')} for m in methods]}),flush=True)
    bi.put(bi.OUT/'blind_compare_summary.json',reports)

if __name__=='__main__':run()
