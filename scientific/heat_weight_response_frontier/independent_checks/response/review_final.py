"""Producer-independent audit of persisted final response artifacts.

Imports only the previous independent rational/interval reviewer. No producer
constructor, estimator, check routine, success report, or resource counter is
imported. Comparison JSON supplies claims and artifact names to be verified.
"""
from collections import Counter
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import gzip,hashlib,importlib.util,json,math,sys
import mpmath as mp

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/response'
spec=importlib.util.spec_from_file_location('independent_prior_bias',ROOT/'independent_checks/statistic/bias_certificate.py')
bc=importlib.util.module_from_spec(spec);spec.loader.exec_module(bc)
I=mp.iv
INPUTS={}


def read(path):
    p=Path(path).resolve();data=p.read_bytes();INPUTS[str(p)]=hashlib.sha256(data).hexdigest()
    return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)


def canonical(word):
    return [(op[0],tuple(op[1]),op[2]) if op[0]=='label' else tuple(op) for op in word]


def inverse(word):
    change={'t':'tdg','tdg':'t','s':'sdg','sdg':'s'}
    return [(change.get(op[0],op[0]),*op[1:]) for op in reversed(word)]


def expand(word,relative):
    out=[]
    for op in canonical(word):
        if op[0]!='ccx':out.append(op);continue
        a,b,t=op[1:]
        if relative:
            out += [('h',t),('t',t),('cx',b,t),('tdg',t),('cx',a,t),('t',t),('cx',b,t),('tdg',t),('h',t)]
        else:
            out += [('h',t),('cx',b,t),('tdg',t),('cx',a,t),('t',t),('cx',b,t),('tdg',t),('cx',a,t),('t',b),('t',t),('h',t),('cx',a,b),('t',a),('tdg',b),('cx',a,b)]
    return out


def mcx(controls,target,scratch):
    if not controls:return [('x',target)]
    if len(controls)==1:return [('cx',controls[0],target)]
    if len(controls)==2:return expand([('ccx',*controls,target)],False)
    wires=[scratch]+list(controls);marks=[True]+[False]*len(controls);raw=[]
    while True:
        possible=[(t,[j for j in range(t+1,len(wires)) if not marks[j]]) for t in range(len(wires)) if marks[t]]
        possible=[(t,avail[:2]) for t,avail in possible if len(avail)>=2]
        if not possible:break
        t,(a,b)=possible[-1];raw.append(('ccx',wires[a],wires[b],wires[t]))
        if t:raw.append(('x',wires[t]))
        marks[t]=False;marks[a]=marks[b]=True
    remaining=[w for w,m in zip(wires,marks) if not m];assert len(remaining)==2
    pre=expand(raw,True)
    return pre+expand([('ccx',*remaining,target)],False)+inverse(pre)


def remap(word,mapping):
    return [(op[0],tuple(mapping[q] for q in op[1]),mapping[op[2]]) if op[0]=='label' else (op[0],*(mapping[q] for q in op[1:])) for op in word]


def ch(a,t):
    return [('sdg',t),('h',t),('tdg',t),('h',t),('s',t),('h',t),('cx',a,t),('h',t),('sdg',t),('h',t),('t',t),('h',t),('s',t)]


def bits(x,reg):return sum(((x>>q)&1)<<i for i,q in enumerate(reg))


def sparse(word,state,labels=None):
    rt=math.sqrt(.5)
    for name,*a in word:
        nxt={}
        for x,v in state.items():
            if name=='h':
                q=a[0];zero=x&~(1<<q);one=zero|(1<<q)
                nxt[zero]=nxt.get(zero,0)+v*rt;nxt[one]=nxt.get(one,0)+v*rt*(-1 if x&(1<<q) else 1);continue
            y=x
            if name=='x':y^=1<<a[0]
            elif name=='cx':
                if x&(1<<a[0]):y^=1<<a[1]
            elif name=='ccx':
                if x&(1<<a[0]) and x&(1<<a[1]):y^=1<<a[2]
            elif name=='label':
                if labels[bits(x,a[0])]<0:y^=1<<a[1]
            elif name in ('z','s','sdg','t','tdg'):
                if x&(1<<a[0]):v*=dict(z=-1,s=1j,sdg=-1j,t=complex(rt,rt),tdg=complex(rt,-rt))[name]
            else:raise AssertionError(name)
            nxt[y]=nxt.get(y,0)+v
        state={x:v for x,v in nxt.items() if abs(v)>1e-14}
    return state


def source_replay(raw):
    if 'source_preparation' not in raw:return F(0),None
    shell=raw['source_shell_bits'];J=len(raw['descriptor']['q_tail']['shell_exponents']);receipts=raw['source_receipt']['rotations']
    scratch=[]
    for name,reg in raw['registers'].items():
        if name.startswith('scratch_'):scratch+=reg
    h=1/I.sqrt(2);ii=I.mpc(0,1);t=I.exp(ii*I.pi/4)
    matrices={'h':I.matrix([[h,h],[h,-h]]),'t':I.matrix([[1,0],[0,t]]),'tdg':I.matrix([[1,0],[0,1/t]]),'x':I.matrix([[0,1],[1,0]]),'z':I.matrix([[1,0],[0,-1]]),'s':I.matrix([[1,0],[0,ii]]),'sdg':I.matrix([[1,0],[0,-ii]])}
    ri=0;total=F(0);checks=[]
    def rotation(theta,target,multiplicity=1):
        nonlocal ri,total
        saved=receipts[ri];ri+=1;U=I.eye(2)
        for gate in saved['word']:U=matrices[gate]*U
        V=I.matrix([[I.exp(-ii*theta/2),0],[0,I.exp(ii*theta/2)]])
        phase=I.exp(ii*I.mpf(saved['comparison_phase']));E=U-phase*V
        bound=bc.upper(I.sqrt(sum(abs(E[i,j])**2 for i in range(2) for j in range(2))))
        declared=F(saved['interval_projective_error']);assert bound<=declared,(float(bound),float(declared))
        assert sum(g in ('t','tdg') for g in saved['word'])==saved['T_count']
        total+=multiplicity*declared;checks.append({'replayed_error_upper':str(bound),'declared':str(declared),'multiplicity':multiplicity})
        return [('sdg',target),('h',target)]+[(g,target) for g in saved['word']]+[('h',target),('s',target)]
    if J&(J-1)==0:
        expected=[('h',q) for q in shell]
    elif J==56:
        a,b,c=shell[5],shell[4],shell[3]
        V=rotation(2*I.atan2(I.sqrt(3),I.mpf(2)),a)
        W=rotation((2*I.atan2(I.mpf(1),I.sqrt(2))-I.pi/2)/2,b,2)
        expected=[('h',q) for q in shell[:3]]+V+[('h',b),('cx',a,b)]+inverse(W)+[('cx',a,b)]+W
        expected += [('h',c)]+expand([('ccx',a,b,scratch[0])],False)+ch(scratch[0],c)+expand([('ccx',a,b,scratch[0])],False)
    else:
        expected=[];prefix=[];remain=J
        for depth,target in enumerate(reversed(shell)):
            remaining=len(shell)-depth;half=1<<(remaining-1)
            if remain==1<<remaining:
                expected += [('h',q) for q in reversed(shell[:remaining])];break
            left=min(remain,half);right=max(0,remain-half)
            if not prefix:
                if right:expected+=rotation(2*I.atan2(I.sqrt(right),I.sqrt(left)),target)
            else:
                expected.append(('h',target));flips=[('x',q) for q,s in prefix if not s];controls=[q for q,s in prefix]
                control=controls[0] if len(controls)==1 else scratch[0]
                conjunction=[] if len(controls)==1 else mcx(controls,control,scratch[1])
                expected+=flips+conjunction
                if right==0:expected+=ch(control,target)+[('h',target),('cx',control,target),('h',target)]
                else:
                    W=rotation((2*I.atan2(I.sqrt(right),I.sqrt(left))-I.pi/2)/2,target,2)
                    expected += [('cx',control,target)]+inverse(W)+[('cx',control,target)]+W
                expected+=inverse(conjunction)+flips
            branch=int(right>0);prefix.append((target,branch))
            if branch:remain=right
    assert ri==len(receipts) and canonical(raw['source_preparation'])==expected
    declared=F(raw['source_receipt'].get('interval_preparation_operator_error','0'));assert total==declared
    psi=sparse(expected,{0:1});ideal={sum(((j>>i)&1)<<q for i,q in enumerate(shell)):1/math.sqrt(J) for j in range(J)}
    overlap=sum(ideal.get(x,0)*a for x,a in psi.items());phase=overlap/abs(overlap)
    distance=math.sqrt(sum(abs(psi.get(x,0)-phase*ideal.get(x,0))**2 for x in set(psi)|set(ideal)))
    assert distance<=float(total)+3e-12
    back=sparse(inverse(expected),psi);inv_error=math.sqrt(sum(abs(v-(1 if x==0 else 0))**2 for x,v in back.items()))
    assert inv_error<1e-11
    return total,{'J':J,'rotation_replays':checks,'source_state_distance':distance,'inverse_error':inv_error,'source_word_exactly_reconstructed':True}


def tables_and_bias(raw,source_error):
    d=raw['descriptor'];cfg=raw['configuration'];method=cfg.get('method','linear');linear=method=='linear'
    degree=1 if linear else cfg['degree'];M=1<<cfg['g'];scale=1<<cfg['p'];shell=d['q_tail']['shell_exponents']
    z=F(raw['normalization_fraction']);mass=F(d['q_tail']['per_side_shell_mass_numerator'],int(d['dyadic_denominator']))
    maximum=F(0);entries=0;negative=0
    def f_derivs(key,index):
        x=I.mpf(1)+bc.interval_fraction(index)/M
        if key=='limit':return [(-1)**q*math.factorial(q)/x**(q+1) for q in range(4)]
        a=I.pi*(1<<int(key))/d['N'];t=I.mpf(0) if int(key)==d['n']-2 and index==M else 1/I.tan(a*x)
        return [a*t,-a*a*(1+t*t),2*a**3*t*(1+t*t),-2*a**4*(1+4*t*t+3*t**4)]
    for key,cells in raw['endpoint_table'].items():
        assert len(cells)==M+1 if linear else len(cells)==M
        for cell,saved in enumerate(cells):
            if linear:
                values=[f_derivs(key,F(cell))[0]-bc.interval_fraction(F(1,8*M*M))];saved=[saved];intervals=[raw['endpoint_intervals'][key][cell]]
            elif method=='nodes':
                y=[f_derivs(key,F(cell)+F(j,3))[0] for j in range(4)]
                values=[y[0],(-5*y[0]+18*y[1]-9*y[2]+2*y[3])/6,(2*y[0]-9*y[1]+18*y[2]-5*y[3])/6,y[3]];intervals=raw['endpoint_intervals'][key][cell]
            else:
                left=f_derivs(key,F(cell));right=f_derivs(key,F(cell+1));m=(degree+1)//2;values=[None]*(degree+1)
                for j in range(m):
                    values[j]=sum(math.comb(j,q)*left[q]/M**q*math.factorial(degree-q)/math.factorial(degree) for q in range(j+1))
                    values[degree-j]=sum(math.comb(j,q)*right[q]/(-M)**q*math.factorial(degree-q)/math.factorial(degree) for q in range(j+1))
                intervals=raw['endpoint_intervals'][key][cell]
            for value,(mag,sg),record in zip(values,saved,intervals):
                assert 0<=mag<=scale and sg in (0,1)
                lo,hi=bc.bounds(value);assert -1<=lo<=hi<=1
                saved_lo,saved_hi=map(F,record)
                assert saved_lo<=lo<=hi<=saved_hi
                e=bc.error(value,F(-mag if sg else mag,scale));assert e<=F(1,scale)
                maximum=max(maximum,e);entries+=1;negative+=sg
    small=[k for k in shell if cfg.get('merge_small_cut') is not None and k<=d['n']-cfg['merge_small_cut']]
    for k in shell:assert str(raw['table_keys'][str(k)])==('limit' if k in small else str(k))
    assert set(raw['endpoint_table'])=={str(v) for v in raw['table_keys'].values()}
    exact=max(shell)+F(d['eta']).denominator.bit_length()-1;assert cfg['exact_fraction_bits']==exact
    lipschitz=F(123370055013617,100000000000000) if method=='nodes' else bc.PI**2/8
    rounding=F(0) if cfg['g']+cfg['round_bits']>=exact else lipschitz/F(1<<(cfg['g']+cfg['round_bits']))
    point=F(1,8*M*M) if linear else F(1,81*M**4) if method=='nodes' else F(1,4**((degree+1)//2)*M**(degree+1))
    factor=F(1) if linear or not raw.get('cell_refinement') else sum((F(M,M+j)**(degree+2) for j in range(M)),F(0))/M
    interpolation=2*mass*point*sum((factor if k>=cfg['g'] else F(1) for k in shell),F(0))
    if raw.get('cell_refinement'):
        ref=raw['cell_refinement'];assert F(ref['uniform_cell_factor'])==factor and F(ref['new_raw_remainder'])==interpolation
    terms={'descriptor':F(d['error']['all_label_raw_bias_upper']),'selector':F(0) if 'source_preparation' in raw else z*F(2*d['descriptor_entries'],1<<cfg['c']),
           'hermite':interpolation,'threshold_quantization':2*len(shell)*mass/scale,'theta_rounding':2*len(shell)*mass*rounding,
           'small_angle_sharing':2*mass*sum((F(8*(1<<(2*k)),d['N']**2) for k in small),F(0))}
    if 'source_preparation' in raw:terms['source_preparation']=2*z*source_error
    if 'bias_components' in raw:assert {k:F(v) for k,v in raw['bias_components'].items()}==terms
    bias=sum(terms.values(),F(0))+F(raw['fixed_coefficient_and_tail_bias']);assert bias==F(raw['raw_bias_fraction'])
    return bias,{'table_controls_checked':entries,'negative_controls':negative,'largest_quantization_error':str(maximum),'recomputed_terms':{k:str(v) for k,v in terms.items()}}


def native_word_check(raw,comp):
    mapping={int(k):v for k,v in comp['original_to_compiled_wire'].items()};B=canonical(raw['B']);phase=canonical(raw['controlled_phase']);N=len(B)
    assert phase[:N]==B and phase[-N:]==list(reversed(B))
    compute=remap(expand(B,True),mapping);core=remap(expand(phase[N:-N],False),mapping)
    assert canonical(comp['compute'])==compute and canonical(comp['compute_inverse'])==inverse(compute)
    assert canonical(comp['controlled_phase'])==compute+core+inverse(compute)
    assert sum(op[0]=='label' for op in core)==1
    seeds=comp['seedbits'];outer=comp['outer_control'];target=comp['phase_target'];scratch=comp['reflection_scratch']
    if 'source_preparation' in raw:
        prep=remap(canonical(raw['source_preparation']),mapping);ordinary=[mapping[q] for q in raw['source_ordinary_bits']]
        assert canonical(comp['source_preparation'])==prep and canonical(comp['initial_preparation'])==[('h',q) for q in ordinary]+prep
    else:prep=[];ordinary=seeds
    reflection=inverse(prep)+[('h',q) for q in ordinary]+[('x',q) for q in seeds]
    reflection += [('x',target),('h',target)]+mcx([outer,*seeds],target,scratch)+[('h',target),('x',target)]
    reflection += [('x',q) for q in reversed(seeds)]+[('h',q) for q in reversed(ordinary)]+prep+[('z',outer)]
    # The uniform compiler may interleave commuting H/X gates per wire.
    if not prep:
        reflection=[]
        for q in seeds:reflection += [('h',q),('x',q)]
        reflection += [('x',target),('h',target)]+mcx([outer,*seeds],target,scratch)+[('h',target),('x',target)]
        for q in reversed(seeds):reflection += [('x',q),('h',q)]
        reflection += [('z',outer)]
    assert canonical(comp['controlled_reflection'])==reflection
    assert canonical(comp['controlled_iterate'])==canonical(comp['controlled_phase'])+reflection
    return {'relative_compute_and_true_adjoint_verified':True,'one_label_query_per_iterate':True,'source_reflection_reconstructed_with_absolute_control_phase':True}


def depths(word,width,oracle_layer=False):
    clocks=[0]*width;tclocks=[0]*width
    for name,*args in word:
        wires=[*args[0],args[1]] if name=='label' else args
        d=max(clocks[q] for q in wires)+(oracle_layer or name!='label');t=max(tclocks[q] for q in wires)+(name in ('t','tdg'))
        for q in wires:clocks[q]=d;tclocks[q]=t
    return max(clocks,default=0),max(tclocks,default=0)


@lru_cache(None)
def qft_counts(m):
    ft=bc.qft(m);gates=3*(m-1)+2*m+m*(m-1)+3*(m//2)
    for j in range(3,m+1):
        item=read(ROOT.parent/f'heat_weight_tight_frontier/results/next/qft/rotation_{j:02d}.json.gz')
        gates+=3*(m-j+1)*len(item['word'])
    return ft['T'],gates,ft['error']


def counts_readout(comp,row,R,bias,z):
    s=row['schedule'];P,K=s['P'],s['K'];m=P.bit_length()-1;assert P==1<<m and K>0 and K%2
    qT,qg,qerr=qft_counts(m);raw_error=bias+bc.statistical(z,P)+z/F(1<<48);failure=bc.median(K)+2*K*qerr+F('.01')
    assert bias<F('.0002') and raw_error<F('.002') and failure<F('.05')
    assert F(s['raw_error'])==raw_error and F(s['failure'])==failure
    word=comp['controlled_iterate'];count=Counter(x[0] for x in word);assert count['ccx']==0 and count['label']==1
    d,t=depths(word,comp['width']);T=count['t']+count['tdg'];calls=K*(P-1)
    init=comp.get('initial_preparation',[('h',q) for q in comp['seedbits']]);ic=Counter(x[0] for x in init);idp,it=depths(init,comp['width']);iT=ic['t']+ic['tdg']
    totals={'T_count':calls*T+K*(qT+iT),'nonoracle_gates':calls*(len(word)-count['label'])+K*(qg+len(init)),
            'T_depth_upper':calls*t+K*(qT+it),'nonoracle_depth_upper':calls*d+K*(qg+idp),
            'label_queries':2*R+calls,'coherent_label_queries':calls,'measured_head_queries':2*R,'allocated_width':comp['width']+m}
    for key,v in totals.items():assert row['totals'][key]==v,(key,row['totals'][key],v)
    ledger=comp['compiled_ledger'];dep,td=depths(word,comp['width'],True)
    assert ledger['T_count']==T and ledger['depth_including_opaque_query_layers']==dep and ledger['T_depth_conservative_scheduled']==td
    assert ledger['nonoracle_elementary_gates']==len(word)-1 and ledger['ordinary_label_queries']==1
    return {'totals_recounted':totals,'decoder_inclusive_raw_error':str(raw_error),'failure':str(failure),'initial_preparation_T_per_run':iT}


DESCRIPTORS={}
def descriptor_check(raw):
    d=raw['descriptor'];key=(d['N'],d['tau'],d['lambda'],d['eta'],len(raw['head_coefficients']))
    if key in DESCRIPTORS:return DESCRIPTORS[key]
    radius=raw['head_preprocessing']['original_descriptor_radius']
    result=bc.check_descriptor(d,raw['head_coefficients'],radius)
    fixed=sum(bc.context(d)['global_bound'](radius).values(),F(0));assert fixed<=F(raw['fixed_coefficient_and_tail_bias'])
    DESCRIPTORS[key]=result;return result


def run(paths):
    reports=[]
    for path in paths:
        comparison=read(path);checked=[]
        for row in comparison['selected']:
            name=row['artifact'];print('Checking '+name,flush=True)
            raw=read(OUT/(name+'_raw.json.gz'));comp=read(OUT/(name+'_word.json.gz'))
            source_error,sourcereport=source_replay(raw);bias,tablereport=tables_and_bias(raw,source_error)
            descriptor=descriptor_check(raw);native=native_word_check(raw,comp)
            resource=counts_readout(comp,row,comparison['R'],bias,F(raw['normalization_fraction']))
            checked.append({'artifact':name,'descriptor':descriptor,'tables_bias':tablereport,'source':sourcereport,'native':native,'resource':resource})
        reports.append({'comparison':str(Path(path).name),'selected_checked':checked})
    decoder=bc.finite_decoder_witness()
    result={'scope':'Saved-artifact audit. No producer constructors, estimators, check routines, success verdicts or counters imported. Previous independent descriptor/QFT rational reviewer reused read-only.',
            'higher_degree_remainder_proof':'For every positive even derivative order q, the cotangent derivative polynomial has nonnegative coefficients when cot(ax)>=0. Its convergent positive cotangent power series subtracts nonnegative qth derivatives from 1/x, so 0<=f^(q)<=q!/x^(q+1). Degree 2m-1 endpoint Hermite remainder has magnitude at most [(x-l)(r-x)]^m/l^(2m+1)<=h^(2m)/(4^m*l^(2m+1)). Finite control range is certified separately by the independently replayed intervals.',
            'finite_decoder':decoder,'comparisons':reports,'input_sha256':INPUTS,'prior_independent_inputs':bc.INPUTS,
            'verdict':'All requested final artifacts passed these checks.'}
    dest=OUT/'review_final_saved_artifacts.json';dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(str(dest),flush=True)


if __name__=='__main__':
    paths=[Path(p) for p in sys.argv[1:]] or sorted(OUT.glob('*_final_comparison.json'))
    if not paths:raise SystemExit('No final comparison artifacts exist; no verdict emitted.')
    run(paths)
