"""Independent complete-program and finite-menu accounting from native words."""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
from collections import Counter
import json
import math
import copy
import review_native_programs as nv
import review_certificates as cv

ROOT=nv.ROOT;OUT=nv.OUT
read=nv.read;digest=nv.digest


def ref(r):
    path=Path(r['path']).resolve();assert digest(path)==r['sha256'];return path


def renaming(c):return dict(T=c['T_count'],gates=c['gates'],depth=c['depth'],T_depth=c['T_depth'])


@lru_cache(None)
def native_qft(path):
    row=read(path);m=row['m'];word=nv.expected_iqft(m)
    assert word==row['word'] and nv.inverse(word)==row['inverse_word']
    cost=renaming(nv.counts(word,m))
    for key in cost:assert cost[key]==row[key]
    receipts={r['j']:r for r in read(OUT/'review_ramps.json')['raw_receipts']}
    error=F(0)
    for j in range(3,m+1):
        source=nv.LIB/f'rotation_{j:02d}.json.gz';rot=read(source)
        assert digest(source)==receipts[j]['sha256']
        error+=3*(m-j+1)*F(rot['interval_operator_upper'])
    assert error==F(row['operator_error'])
    for rec in row['rotation_receipts']:
        if rec['j']>=3:assert digest(rec['path'])==rec['sha256']
        else:assert rec==dict(j=2,kind='exact T')
    return cost,error


@lru_cache(None)
def native_clock(path):
    checked=nv.clock(str(path));return renaming(checked),F(checked['operator_error'])


@lru_cache(None)
def body(path):
    comp=read(path);word=comp['controlled_iterate'];width=comp['width']
    d=[0]*width;t=[0]*width;counts=Counter(op[0] for op in word)
    assert counts['label']==1
    for op in word:
        wires=([*op[1],op[2]] if op[0]=='label' else op[1:])
        assert all(isinstance(q,int) and 0<=q<width for q in wires)
        end=max(d[q] for q in wires)+int(op[0]!='label')
        tend=max(t[q] for q in wires)+int(op[0] in ('t','tdg'))
        for q in wires:d[q]=end;t[q]=tend
    cost=dict(T=counts['t']+counts['tdg'],gates=len(word)-1,T_depth=max(t),depth=max(d),width=width)
    initial=renaming(nv.counts(comp['initial_preparation'],width))
    return comp,cost,initial


def binomial(K,p):
    return sum((F(math.comb(K,k))*p**k*(1-p)**(K-k) for k in range(K//2+1,K+1)),F(0))


@lru_cache(None)
def certificate(path,Z,beta):return cv.review(Path(path),(F(Z),F(beta)))


def review_program(path,override=None):
    p=read(path) if override is None else override
    P,S,K=p['P'],p['S'],p['K'];route=p['route']
    assert P>=2 and P&(P-1)==0 and S in (1,2,4,8,16) and K in range(1,128,2)
    assert route in ('randomized','padded')
    m=P.bit_length()-1;s=S.bit_length()-1;phasewidth=m+(s if route=='padded' else 0)
    contract=p['contract']
    assert [F(contract[k]) for k in ('raw_margin','bias_cap','total_failure','logical_reserve')]==[F(1,500),F(1,5000),F(1,20),F(1,100)]
    ob=p['observable'];Z,beta=F(ob['Z']),F(ob['bias'])
    assert Z>0 and 0<=beta<=F(1,5000)
    assert digest(ob['raw_path'])==ob['raw_sha256'] and digest(ob['native_path'])==ob['native_sha256']
    bound=read(ref(p['input_binding']))
    if 'observable' in bound:assert ob==bound['observable']
    else:
        assert F(bound['Z'])==Z and F(bound['bias'])==beta
        assert bound['totals']['measured_head_queries']==ob['head_labels']
    raw=read(ob['raw_path'])
    assert F(raw['normalization_fraction'])==Z and F(raw['raw_bias_fraction'])==beta
    assert len(raw['head_coefficients'])==ob['head_labels']
    assert raw['head_preprocessing']['head_label_queries']==ob['head_labels']
    comp,it,init=body(Path(ob['native_path']))
    assert it==p['iterate_profile'] and init==p['initial_profile']
    width=comp['width'];assert p['phase_wires']==list(range(width,width+phasewidth))
    assert p['clock_local_wires']==list(range(m))
    assert p['initially_zero_padding_local_wires']==list(range(m,phasewidth))
    assert p['powers']==[dict(control_wire=width+k,repeat=2**k,outer_port=comp['outer_control']) for k in range(m)]
    assert sum(row['repeat'] for row in p['powers'])==P-1
    clockpath,qftpath=ref(p['clock']),ref(p['inverse_qft'])
    assert read(clockpath)['P']==P and read(qftpath)['m']==phasewidth
    c,ec=native_clock(clockpath);q,eq=native_qft(qftpath)
    if route=='randomized':
        ramp_path=ref(p['ramp']);r=read(ramp_path)
        assert (r['m'],r['P'],r['S'])==(m,P,S)
        w=r['worst'];mean={k:F(v) for k,v in r['mean'].items()};er=F(r['worst_operator_error'])
        # The separate fresh-context checker must cover exactly this file.
        ramp_report=read(OUT/'review_ramps.json')
        rows=ramp_report['ramps']
        match=[x for x in rows if (ROOT/x['file']).resolve()==ramp_path.resolve()]
        assert len(match)==1 and match[0]['sha256']==digest(ramp_path)
    else:
        assert p['ramp'] is None
        w=dict(T=0,gates=0,T_depth=0,depth=0);mean={k:F(0) for k in w};er=F(0)
    assert p['randomization']['classical_random_bits_per_run']==(s if route=='randomized' else 0)
    assert p['randomization']['branches_are_alternatives_not_repetitions'] is True
    if route=='randomized':assert 'freshly sampled EACH repetition' in p['randomization']['rule']
    assert p['decoder']['callable']=='decision'
    assert ref(p['decoder']['wrapper'])==ROOT/'src/random_sign/native.py'
    assert ref(p['decoder']['accepted'])==nv.ACCEPTED/'src/statistic/finite_decoder.py'
    certpath=ref(p['certificate']);cert=certificate(str(certpath),str(Z),str(beta));rawcert=read(certpath)['certificate']
    assert (rawcert['P'],rawcert['S'])==(P,S)
    prob=F(cert['statistical_upper']);assert 0<=prob<F(1,2)
    stat=binomial(K,prob);synth=2*K*(ec+eq+er);failure=stat+synth+F(1,100)
    error=p['error']
    expected=dict(raw_bias=beta,Z=Z,decoder=F(1,2**48),single_statistical=prob,statistical=stat,
                  clock_operator=ec,qft_operator=eq,ramp_operator_worst=er,synthesis_failure=synth,logical_reserve=F(1,100),total=failure)
    for key,value in expected.items():assert F(error[key])==value
    assert error['feasible'] is True and failure<F(1,20)
    assert all(binomial(k,prob)+2*k*(ec+eq+er)+F(1,100)>=F(1,20) for k in range(1,K,2))
    base={k:(P-1)*it[k]+init[k]+c[k]+q[{'T_depth':'T','depth':'gates'}.get(k,k)] for k in ('T','gates','T_depth','depth')}
    worst={k:K*(base[k]+w[k]) for k in base};average={k:K*(base[k]+mean[k]) for k in base}
    resources=p['resources'];assert resources['worst']==worst
    assert {k:F(v) for k,v in resources['mean'].items()}==average
    required=dict(head_queries=ob['head_labels'],coherent_queries=K*(P-1),total_queries=ob['head_labels']+K*(P-1),longest_chain=P-1,
                  single_run_worst_T_depth=base['T_depth']+w['T_depth'],single_run_worst_depth=base['depth']+w['depth'],allocated_width=width+phasewidth,
                  classical_random_bits=K*s if route=='randomized' else 0,random_branch_count=S if route=='randomized' else 1)
    for key,value in required.items():assert resources[key]==value
    assert resources['readout_only']==dict(clock=c,qft=q,ramp_worst=w,ramp_mean={k:str(v) for k,v in mean.items()})
    baseline=bound.get('resources')
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path),tag=p['tag'],response=p['response'],route=route,P=P,S=S,K=K,
                certificate_sha256=digest(certpath),failure=str(failure),worst=worst,mean={k:str(v) for k,v in average.items()},
                **required,baseline_resources=baseline)


def menu_probability(tag,response,P,S,Z,beta,oldrows):
    key=(tag,response,P,S)
    if key in oldrows:return F(oldrows[key]['statistical_failure'])
    gap=(F(1,500)-beta)/Z-F(1,2**48)
    if P<32 or P*gap/(2*cv.HI)<=F(3,4):return F(1)
    path=OUT/f'cert_{tag}_{response}_P{P}_S{S}.json'
    checked=certificate(str(path),str(Z),str(beta))
    return F(checked['statistical_upper'])


def attempt_cost(p,P,S,route):
    ob=p['observable'];_,it,init=body(Path(ob['native_path']))
    m=P.bit_length()-1;s=S.bit_length()-1
    clockpath=nv.ACCEPTED/f'results/sign/blind_clock_m{m:02d}.json.gz'
    if not clockpath.exists():clockpath=OUT/f'native/blind_clock_m{m:02d}.json.gz'
    c,ec=native_clock(clockpath);q,eq=native_qft(OUT/f'native/iqft_m{m+(s if route=="padded" else 0):02d}.json.gz')
    rampT=0;er=F(0)
    if route=='randomized':
        r=read(OUT/f'native/ramp_m{m:02d}_S{S}.json.gz')
        rampT=r['worst']['T'];er=F(r['worst_operator_error'])
    return (P-1)*it['T']+init['T']+c['T']+q['T']+rampT,ec+eq+er


def review_resource_menu(program_rows):
    source=OUT/'resource_menu.json';menu=read(source)['rows']
    oldboundpath=OUT/'bound_menu.json';proof=read(OUT/'review_menu.json')
    assert digest(oldboundpath)==proof['sha256'] and proof['status']=='passed'
    old={(r['tag'],r['response'],r['P'],r['S']):r for r in read(oldboundpath)['rows']}
    selected={(r['tag'],r['response'],r['route'],r['S']):r for r in program_rows}
    keys={(r['tag'],r['response'],r['route'],r['S']) for r in menu}
    assert keys==set(selected) and len(keys)==len(menu)
    rows=[]
    for entry in menu:
        tag,response,route,S=(entry[k] for k in ('tag','response','route','S'))
        chosen=selected[tag,response,route,S];program=read(ROOT/chosen['path'])
        Z,beta=F(program['observable']['Z']),F(program['observable']['bias'])
        _,bodycost,_=body(Path(program['observable']['native_path']))
        attempts=entry['attempts'];assert [a['P'] for a in attempts]==[2**m for m in range(1,19)]
        incumbent=None;witness=None;caps=pruned=feasible=0
        for a in attempts:
            P=a['P'];reason=a.get('reason','')
            if reason=='iterate-only T lower bound exceeds feasible incumbent':
                lower=(P-1)*bodycost['T']
                assert incumbent is not None and lower==a['lower_T'] and lower>=incumbent
                pruned+=1;continue
            prob=menu_probability(tag,response,P,S,Z,beta,old)
            if reason=='certificate >=1/2; no helpful majority':
                assert F(a['statistical'])==prob and prob>=F(1,2);caps+=1;continue
            per_run,error=attempt_cost(program,P,S,route)
            feasible_K=[K for K in range(1,128,2) if binomial(K,prob)+2*K*error+F(1,100)<F(1,20)]
            if reason=='no feasible K including synthesis':
                assert not feasible_K and F(a['single_statistical'])==prob;continue
            assert reason=='' and feasible_K
            K=feasible_K[0];cost=K*per_run;failure=binomial(K,prob)+2*K*error+F(1,100)
            assert K==a['K'] and cost==a['T'] and failure==F(a['failure'])
            feasible+=1
            if incumbent is None or cost<incumbent:incumbent=cost;witness=(P,K)
        assert incumbent==chosen['worst']['T'] and witness==(chosen['P'],chosen['K'])
        rows.append(dict(tag=tag,response=response,route=route,S=S,phase_sizes=18,probability_caps=caps,
                         lower_bound_prunes=pruned,fully_charged_candidates=feasible,selected_P=witness[0],selected_K=witness[1],T=incumbent))
    result=dict(status='passed',sha256=digest(source),menu_rows=len(rows),entries=18*len(rows),rows=rows,
                scope='Every declared P checked: either certified probability cap, exact first feasible odd K with charged native modules, or iterate-only T lower bound against an already feasible incumbent. Selection is optimal only within these certificates and menus.')
    (OUT/'review_resource_menu.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='rows'}


def main():
    cv.side_conditions()
    paths=sorted((OUT/'programs').glob('*.json'));assert paths
    rows=[review_program(path) for path in paths]
    controlpath=OUT/'programs/discovery_n68_nodes_randomized_S4.json';original=read(controlpath)
    rejected=[]
    for kind in ('power','query_cost','failure','fresh_randomness','native_hash','bias'):
        bad=copy.deepcopy(original)
        if kind=='power':bad['powers'][-1]['repeat']*=2
        elif kind=='query_cost':bad['resources']['total_queries']-=1
        elif kind=='failure':bad['error']['synthesis_failure']='0'
        elif kind=='fresh_randomness':bad['randomization']['rule']='Reuse one random branch for every repetition'
        elif kind=='native_hash':bad['ramp']['sha256']='0'*64
        elif kind=='bias':bad['observable']['bias']='0'
        try:review_program(controlpath,bad)
        except AssertionError:rejected.append(kind)
        else:raise AssertionError('Corruption accepted: '+kind)
    result=dict(status='passed',program_count=len(rows),rows=rows,negative_controls_rejected=rejected,
                scope='Independent literal body/initial/clock/IQFT/ramp resource reconstruction, hashes, complete phase and repetition schedules, exact certified statistics and synthesis allowances, fresh-z and decoder bindings. Accepted observable correctness is retained by byte-identical source binding.')
    (OUT/'review_program_ledgers.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    print(json.dumps(review_resource_menu(rows)))


if __name__=='__main__':main()
