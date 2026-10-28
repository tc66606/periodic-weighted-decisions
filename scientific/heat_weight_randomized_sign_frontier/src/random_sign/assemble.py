"""Complete same-observable finite programs for randomized and padded readout.

The finite resource menu is exhaustive within its declared P,S,K values. A
candidate may be excluded before clock synthesis only by a strict iterate-only
T lower bound exceeding an already fully charged feasible schedule for that S.
"""
from fractions import Fraction as F
from pathlib import Path
from collections import Counter
import csv,json,sys
from native import ROOT,OLD,OUT,read,save,sha,ref,clock,qft,ramps,ledger
from discrete_bound import certificate,majority_bound,P_MENU,S_MENU,K_MENU,construction
RESULT=ROOT/'results/random_sign'

def candidates():
    rows=read(OLD/'results/sign/final_comparison.json')['rows']
    selected=[r for r in rows if r['route']=='sine_sign' and r['tag'] in ('discovery_n12','discovery_n68')]
    for r in selected:
        p=read(OLD/r['program']);r=dict(r);r['source_program']=ref(OLD/r['program']);r['input_program']=p
        yield r
    fresh=RESULT/'confirmation/index.json'
    if fresh.exists():
        for r in read(fresh)['rows']:
            name=r['artifact'];folder=fresh.parent
            yield dict(tag=r['tag'],response=r['method'],Z=r['Z'],bias=r['bias'],baseline=r['totals'],
                input_program=dict(observable=dict(raw_path=str(folder/(name+'_raw.json.gz')),raw_sha256=sha(folder/(name+'_raw.json.gz')),
                    native_path=str(folder/(name+'_word.json.gz')),native_sha256=sha(folder/(name+'_word.json.gz')),
                    head_labels=r['totals']['measured_head_queries'],Z=r['Z'],bias=r['bias'])),source_program=ref(folder/(name+'.json')))

def body_profile(ob):
    comp=read(ob['native_path']);w=comp['controlled_iterate'];cnt=Counter(o[0] for o in w)
    d=[0]*comp['width'];t=[0]*comp['width']
    for o in w:
        bits=[*o[1],o[2]] if o[0]=='label' else o[1:]
        dd=max(d[i] for i in bits)+(o[0]!='label');tt=max(t[i] for i in bits)+(o[0] in ('t','tdg'))
        for i in bits:d[i]=dd;t[i]=tt
    assert cnt['label']==1
    return comp,dict(T=cnt['t']+cnt['tdg'],gates=len(w)-1,T_depth=max(t),depth=max(d),width=comp['width'])

def program(r,P,S,K,route,cert):
    ob=r['input_program']['observable'];comp,body=body_profile(ob)
    init=ledger(comp['initial_preparation'],comp['width']);m=P.bit_length()-1;s=S.bit_length()-1
    cw,cr=clock(m);qw,qr=qft(m if route=='randomized' else m+s)
    if route=='randomized':
        ramp,rr=ramps(m,S);worst=ramp['worst'];mean={k:F(v) for k,v in ramp['mean'].items()};er=F(ramp['worst_operator_error'])
    else:rr=None;worst=dict(T=0,gates=0,depth=0,T_depth=0);mean={k:F(0) for k in worst};er=F(0)
    ec=F(cw['operator_error']);eq=F(qw['operator_error']);p=F(cert['statistical_failure']);stat=majority_bound(p,K)
    fail=stat+2*K*(ec+eq+er)+F('.01')
    single=dict(T=(P-1)*body['T']+init['T']+cw['T']+qw['T'],
        gates=(P-1)*body['gates']+init['gates']+cw['gates']+qw['gates'],
        T_depth=(P-1)*body['T_depth']+init['T_depth']+cw['T_depth']+qw['T'],
        depth=(P-1)*body['depth']+init['depth']+cw['depth']+qw['gates'])
    worst_total={k:K*(single[k]+worst[k]) for k in single}
    mean_total={k:str(K*(single[k]+mean[k])) for k in single}
    phase_width=m+(s if route=='padded' else 0)
    head=ob['head_labels'];cn=RESULT/f"cert_{r['tag']}_{r['response']}_P{P}_S{S}.json"
    # Every emitted selected program includes a full closed-cell certificate.
    raw_cert=certificate(P,S,r['Z'],r['bias'],True);save(cn,dict(construction=construction(),certificate=raw_cert))
    return dict(schema='finite_randomized_sign_program_v1',tag=r['tag'],response=r['response'],route=route,
        P=P,S=S,K=K,contract=dict(raw_margin='.002',bias_cap='.0002',total_failure='.05',logical_reserve='.01',access='same unknown XOR binary-label oracle'),
        observable=ob,input_binding=r['source_program'],iterate_profile=body,initial_profile=init,
        randomization=dict(classical_random_bits_per_run=s if route=='randomized' else 0,
            rule='Uniform independent z in0..S-1, freshly sampled EACH repetition' if route=='randomized' else 'No classical offset drawn; measure padded outcome k=S*j+z',
            branches_are_alternatives_not_repetitions=True),clock=cr,ramp=rr,inverse_qft=qr,
        phase_wires=list(range(comp['width'],comp['width']+phase_width)),clock_local_wires=list(range(m)),
        initially_zero_padding_local_wires=list(range(m,phase_width)),
        ordered_operations=['prepare source','prepare low-m sine clock','apply selected negative ramp (randomized only)','controlled powers on low-m phase bits only','inverse QFT on all phase bits','measure','finite shifted decode and classical median'],
        powers=[dict(control_wire=comp['width']+j,repeat=1<<j,outer_port=comp['outer_control']) for j in range(m)],
        inverse_semantics='Accepted literal compute inverse and actual-source reflection unchanged; random ramp adjoint is literal reversed native word.',
        decoder=dict(wrapper=ref(ROOT/'src/random_sign/native.py'),callable='decision',accepted=ref(OLD/'src/statistic/finite_decoder.py'),
            rule='signed_mean(S*j+z,S*P,bits48); padded outcome k becomes (k//S,k%S); exact same dyadic head; positive tie, ties charged as failures'),
        certificate=ref(cn),error=dict(raw_bias=r['bias'],Z=r['Z'],decoder='1/281474976710656',single_statistical=str(p),
            statistical=str(stat),clock_operator=str(ec),qft_operator=str(eq),ramp_operator_worst=str(er),
            synthesis_failure=str(2*K*(ec+eq+er)),logical_reserve='.01',total=str(fail),feasible=fail<F('.05')),
        resources=dict(worst=worst_total,mean=mean_total,head_queries=head,coherent_queries=K*(P-1),total_queries=head+K*(P-1),
            longest_chain=P-1,single_run_worst_T_depth=single['T_depth']+worst['T_depth'],single_run_worst_depth=single['depth']+worst['depth'],
            allocated_width=comp['width']+phase_width,optimal_active_width=None,classical_random_bits=K*s if route=='randomized' else 0,
            random_branch_count=S if route=='randomized' else 1,readout_only=dict(clock=ledger(cw['word'],m),qft=ledger(qw['word'],phase_width),ramp_worst=worst,ramp_mean={k:str(v) for k,v in mean.items()})),
        cost_scope='True native modules and explicit finite repetition loops. Serial composition; QFT serialized for matched depth controls. Initial zero padding free reset convention same as baseline; allocated wires charged. No oracle-interior/physical cost. No claim of optimal active-qubit schedule.')

def run():
    rows=[];menus=[]
    for r in candidates():
        comp,body=body_profile(r['input_program']['observable'])
        for route in ('randomized','padded'):
            for S in S_MENU:
                best=None;attempts=[]
                for P in P_MENU:
                    c=certificate(P,S,r['Z'],r['bias'])
                    p=F(c['statistical_failure'])
                    if p>=F(1,2):
                        attempts.append(dict(P=P,reason='certificate >=1/2; no helpful majority',statistical=str(p)));continue
                    if best is not None and (P-1)*body['T']>=best['resources']['worst']['T']:
                        attempts.append(dict(P=P,reason='iterate-only T lower bound exceeds feasible incumbent',lower_T=(P-1)*body['T']));continue
                    # Exact minimal K: every extra K increases cost and synthesis charge.
                    m=P.bit_length()-1;clk,_=clock(m);ft,_=qft(m+(S.bit_length()-1 if route=='padded' else 0))
                    ramp,_=ramps(m,S) if route=='randomized' else ({'worst_operator_error':'0'},None)
                    err=F(clk['operator_error'])+F(ft['operator_error'])+F(ramp['worst_operator_error'])
                    feasible=[]
                    for K in K_MENU:
                        f=majority_bound(p,K)+2*K*err+F('.01')
                        if f<F('.05'):feasible.append(K);break
                    if not feasible:
                        attempts.append(dict(P=P,reason='no feasible K including synthesis',single_statistical=str(p)));continue
                    candidate=program(r,P,S,feasible[0],route,c)
                    assert candidate['error']['feasible']
                    attempts.append(dict(P=P,K=feasible[0],T=candidate['resources']['worst']['T'],failure=candidate['error']['total']))
                    if best is None or candidate['resources']['worst']['T']<best['resources']['worst']['T']:best=candidate
                assert best is not None
                name=f"{r['tag']}_{r['response']}_{route}_S{S}"
                path=save(RESULT/'programs'/f'{name}.json',best)
                rows.append(dict(tag=r['tag'],response=r['response'],route=route,S=S,P=best['P'],K=best['K'],Z=r['Z'],bias=r['bias'],
                    failure=best['error']['total'],program=ref(path),resources=best['resources'],baseline=r['baseline']))
                menus.append(dict(tag=r['tag'],response=r['response'],route=route,S=S,attempts=attempts))
                print(name,best['P'],best['K'],best['resources']['worst']['T'],flush=True)
    save(RESULT/'final_comparison.json',dict(starting_commit='04df314c0b82499bfd59d640749d5f7f31f569ce',rows=rows,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/random_sign/native.py',ROOT/'src/random_sign/discrete_bound.py']},
        selection='Minimal worst-branch T over finite P powers2 m1..18,oddK1..127 for each S and route; no general optimality claim'))
    save(RESULT/'resource_menu.json',dict(rows=menus))
    keys=['tag','response','route','S','P','K','Z','bias','statistical_failure','synthesis_failure','failure','queries','longest_chain','T_worst','T_mean','T_depth_worst','nonoracle_depth_worst','width','random_bits','program','certificate']
    with (RESULT/'resources.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in rows:
            p=read(r['program']['path']);a=p['resources'];e=p['error']
            writer.writerow(dict(tag=r['tag'],response=r['response'],route=r['route'],S=r['S'],P=r['P'],K=r['K'],Z=r['Z'],bias=r['bias'],
                statistical_failure=e['statistical'],synthesis_failure=e['synthesis_failure'],failure=e['total'],queries=a['total_queries'],
                longest_chain=a['longest_chain'],T_worst=a['worst']['T'],T_mean=a['mean']['T'],T_depth_worst=a['worst']['T_depth'],
                nonoracle_depth_worst=a['worst']['depth'],width=a['allocated_width'],random_bits=a['classical_random_bits'],program=r['program']['path'],certificate=p['certificate']['path']))
if __name__=='__main__':run()
