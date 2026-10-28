"""Independent equal-shell source state, paired reflection and literal costs.

The public measured-head tail has identical masses on every side-shell.
Replace its floor-CDF seed by a uniform shell superposition. For J=56 use
|uniform_7> tensor |+>**3; for J=4 use only Hadamards. Synthesis error changes
the represented fixed score once; the exact synthesized preparation and its
literal adjoint implement the source reflection on every iterate.
"""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import copy, json, math, random, sys
import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/statistic'))
import blind_compiled as api
from coherent_compile import compile_bundle, kg_mcx, remap_word
from shared_observable.boolean_circuit import inverse, ledger, classical_apply, sparse_apply
bi,ml=api.bi,api.ml
_ROTATION_CACHE={}


def interval_rz_error(word,theta):
    """Outward projective operator bound; analogue of accepted QFT replay."""
    mp.mp.dps=80;mp.iv.dps=75
    I,J=mp.iv.mpf,mp.iv.mpc
    h=1/mp.iv.sqrt(2);t=mp.iv.exp(J(0,mp.iv.pi/4))
    gates={'h':[[h,h],[h,-h]],'t':[[1,0],[0,t]],'tdg':[[1,0],[0,1/t]],
           's':[[1,0],[0,J(0,1)]],'sdg':[[1,0],[0,J(0,-1)]],
           'x':[[0,1],[1,0]],'z':[[1,0],[0,-1]]}
    U=mp.iv.matrix([[1,0],[0,1]]);radius=I(0)
    for start in range(0,len(word),16):
        B=mp.iv.matrix([[1,0],[0,1]])
        for g in word[start:start+16]:B=mp.iv.matrix(gates[g])*B
        U=B*U
        C=mp.iv.matrix([[J(float(U[a,b].real.mid),float(U[a,b].imag.mid)) for b in range(2)] for a in range(2)])
        radius+=mp.iv.sqrt(sum(abs(U[a,b]-C[a,b])**2 for a in range(2) for b in range(2)))
        U=C
    V=mp.iv.matrix([[mp.iv.exp(J(0,-theta/2)),0],[0,mp.iv.exp(J(0,theta/2))]])
    mid=lambda z:complex(float(z.real.mid),float(z.imag.mid))
    overlap=sum(np.conj(mid(V[k,k]))*mid(U[k,k]) for k in range(2))
    phase=math.atan2(overlap.imag,overlap.real)
    E=U-mp.iv.exp(J(0,I(phase)))*V
    upper=mp.mpf((mp.iv.sqrt(sum(abs(E[a,b])**2 for a in range(2) for b in range(2)))+radius)._mpi_[1])
    return Fraction(str(mp.nstr(upper*mp.mpf('1.0000000000000000000001'),65))),phase


def synthesized_ry(theta,theta_iv,target,epsilon=1e-10):
    from qiskit.synthesis import gridsynth_rz
    key=(mp.nstr(theta,75),epsilon)
    if key not in _ROTATION_CACHE:
        circuit=gridsynth_rz(float(theta),epsilon)
        bare=[x.operation.name for x in circuit.data]
        error,phase=interval_rz_error(bare,theta_iv)
        assert error < Fraction(1,10**8)
        _ROTATION_CACHE[key]=(bare,error,phase)
    bare,error,phase=_ROTATION_CACHE[key]
    word=[('sdg',target),('h',target)]+[(g,target) for g in bare]+[('h',target),('s',target)]
    return word,dict(word=bare,T_count=sum(g in ('t','tdg') for g in bare),
                     interval_projective_error=str(error),comparison_phase=phase,
                     angle=str(theta),requested_epsilon=epsilon)


def controlled_h(control,target):
    # Exact CH, identical phase convention to accepted controlled_word.
    return [('sdg',target),('h',target),('tdg',target),('h',target),('s',target),
            ('h',target),('cx',control,target),('h',target),
            ('sdg',target),('h',target),('t',target),('h',target),('s',target)]


def generic_source_word(shellbits,scratch,J):
    """Uniform 0..J-1 using one partial binary prefix per level.

    All complete subtrees use Hadamards. There is at most one incomplete
    subtree on each level, with a conditional Ry determined by its two child
    counts. Negative prefix controls are complemented reversibly. Synthesis
    errors telescope over at most log2(J) rotations; each controlled rotation
    uses a synthesized half-angle and its literal adjoint, so its projective
    phase cancels exactly. Prefix conjunction work is uncomputed.
    """
    from coherent_compile import expand
    mp.mp.dps=80;mp.iv.dps=75
    if J<1 or J>(1<<len(shellbits)):raise ValueError(J)
    word=[];receipts=[];error=Fraction(0);prefix=[];remain=J
    for depth,target in enumerate(reversed(shellbits)):
        remaining=len(shellbits)-depth;half=1<<(remaining-1)
        if remain==1<<remaining:
            word += [('h',q) for q in reversed(shellbits[:remaining])]
            break
        left=min(remain,half);right=max(0,remain-half)
        if not prefix:
            if right:
                theta=2*mp.atan(mp.sqrt(mp.mpf(right)/left))
                theta_iv=2*mp.iv.atan2(mp.iv.sqrt(right),mp.iv.sqrt(left))
                V,rec=synthesized_ry(theta,theta_iv,target)
                word+=V;receipts.append(rec);error+=Fraction(rec['interval_projective_error'])
            # The top branch is deterministic zero when J is smaller than
            # half; minimal shell register lengths normally avoid this case.
        else:
            word.append(('h',target))
            flips=[('x',q) for q,value in prefix if not value]
            controls=[q for q,_ in prefix]
            if len(controls)==1:
                conjunction=[];control=controls[0]
            else:
                control=scratch[0]
                conjunction=kg_mcx(controls,control,scratch[1],True)
            word+=flips+conjunction
            if right==0:
                # Ry(-pi/2)=Z H, both factors controlled exactly.
                word+=controlled_h(control,target)+[('h',target),('cx',control,target),('h',target)]
            else:
                delta=(2*mp.atan(mp.sqrt(mp.mpf(right)/left))-mp.pi/2)/2
                delta_iv=(2*mp.iv.atan2(mp.iv.sqrt(right),mp.iv.sqrt(left))-mp.iv.pi/2)/2
                W,rec=synthesized_ry(delta,delta_iv,target)
                word += [('cx',control,target)]+inverse(W)+[('cx',control,target)]+W
                receipts.append(rec);error+=2*Fraction(rec['interval_projective_error'])
            word+=inverse(conjunction)+flips
        branch=int(right>0);prefix.append((target,branch))
        if branch:remain=right
    return word,error,dict(kind='generic truncated-uniform binary source',rotations=receipts,
                          interval_preparation_operator_error=str(error),
                          preparation_T=sum(o[0] in ('t','tdg') for o in word),
                          theorem='At most ceil(log2 J) rotations; one incomplete prefix per binary level. O(log^2 J) exact prefix control gates, unchanged normalization.')


def source_word(shellbits,scratch,J):
    """Prepare shells 0..J-1. All unsupported invalid states remain harmless."""
    if J and J&(J-1)==0:
        return [('h',q) for q in shellbits],Fraction(0),dict(kind='power-of-two shell source',rotations=[])
    if J!=56:return generic_source_word(shellbits,scratch,J)
    scratch=scratch[0]
    lo=shellbits[:3];a,b,c=shellbits[5],shellbits[4],shellbits[3]
    mp.mp.dps=80;mp.iv.dps=75
    theta=2*mp.atan(mp.sqrt(3)/2)
    theta_iv=2*mp.iv.atan2(mp.iv.sqrt(3),mp.iv.mpf(2))
    V,rv=synthesized_ry(theta,theta_iv,a)
    delta=(2*mp.atan(1/mp.sqrt(2))-mp.pi/2)/2
    delta_iv=(2*mp.iv.atan2(mp.iv.mpf(1),mp.iv.sqrt(2))-mp.iv.pi/2)/2
    W,rw=synthesized_ry(delta,delta_iv,b)
    # When a=0, W and its literal adjoint cancel. When a=1, the
    # sequence realizes Ry(2*delta), up to an error <= 2*err(W).
    word=[('h',q) for q in lo]+V+[('h',b)]+[('cx',a,b)]+inverse(W)+[('cx',a,b)]+W
    # c is uniform except prefix ab=11, whose valid child is c=0.
    word += [('h',c),('ccx',a,b,scratch)]+controlled_h(scratch,c)+[('ccx',a,b,scratch)]
    # Expand the two exact Toffolis in the coherent source preparation.
    from coherent_compile import expand
    word=expand(word,relative=False)
    error=Fraction(rv['interval_projective_error'])+2*Fraction(rw['interval_projective_error'])
    return word,error,dict(kind='uniform7 tensor uniform8',rotations=[rv,rw],
                          interval_preparation_operator_error=str(error),
                          preparation_T=Counter(o[0] for o in word)['t']+Counter(o[0] for o in word)['tdg'])


def transform(b,d):
    """Transform either linear or Bernstein sampler; returns a new dictionary.

    The sampler must use the accepted measured-head address prefix: an old
    seed_select register, one-hot shell_k predicates, and coordinate register.
    Its later response and phase operations are retained literally.
    """
    assert not d['head']
    out=copy.deepcopy(b);qc=out['qc'];oldB=out['B'];regs=qc.regs
    shell_exponents=list(d['q_tail']['shell_exponents']);J=len(shell_exponents)
    assert J>=1
    coordinate=set(regs['coordinate'])
    def touches(op):return any(q in coordinate for q in op[1:])
    prefix_end=next(i for i,op in enumerate(oldB) if touches(op))
    ordinal=qc.alloc('equal_shell_source',(J-1).bit_length())
    side=qc.bit('equal_side_source');root=qc.bit('equal_shell_root')
    stack=qc.alloc('equal_shell_decoder',len(ordinal))
    qc.ops=[]
    qc.gate('x',regs['tail'][0]);qc.gate('cx',side,regs['positive_side'][0]);qc.gate('cx',side,regs['public_sign'][0])
    qc.gate('x',root)
    def visit(depth,prefix,parent):
        if depth==len(ordinal):
            if prefix<J:qc.gate('cx',parent,regs['shell_'+str(shell_exponents[prefix])][0])
            return
        remaining=len(ordinal)-depth-1
        if prefix*(1<<(remaining+1))>=J:return
        bit=ordinal[remaining];child=stack[depth]
        qc.gate('x',bit);qc.gate('ccx',parent,bit,child);qc.gate('x',bit)
        visit(depth+1,2*prefix,child)
        qc.gate('cx',parent,child)
        visit(depth+1,2*prefix+1,child)
        qc.gate('cx',parent,child)
        qc.gate('x',bit);qc.gate('ccx',parent,bit,child);qc.gate('x',bit)
    visit(0,0,root);qc.gate('x',root)
    source_selector=qc.ops.copy()
    out['B']=source_selector+oldB[prefix_end:]
    old_select=set(regs['seed_select'])
    out['seeds']=[q for q in out['seeds'] if q not in old_select]+ordinal+[side]
    prep,err,receipt=source_word(ordinal,qc.scratch,J)
    out['source_preparation']=prep;out['source_shell_bits']=ordinal
    out['source_ordinary_bits']=[q for q in out['seeds'] if q not in ordinal]
    out['source_error']=err;out['source_receipt']=receipt
    out['source_scratch_bits']=[q for q in qc.scratch if any(q in op[1:] for op in prep)]
    c=out['config']['c'];old_selector_bias=out['Z']*Fraction(2*d['descriptor_entries'],1<<c)
    out['bias']=out['bias']-old_selector_bias+2*out['Z']*err
    if 'bias_components' in out:
        out['bias_components']['selector']=Fraction(0)
        out['bias_components']['source_preparation']=2*out['Z']*err
        assert sum(out['bias_components'].values(),Fraction(0))==out['bias']
    if 'selector' not in out.get('chunks',{}):
        address_end=out['chunks']['address']['macro_gates']
        out['chunks']['address']=bi.counts(oldB[prefix_end:address_end])
    out['chunks']['selector']=bi.counts(source_selector)
    out['config']['source']='synthesized equal-shell state with exact paired reflection'
    out['source_law']=dict(ideal_shell_probability=str(Fraction(1,J)),ideal_side_probability='1/2',
                           represented_law='Squared amplitudes of the persisted synthesized source word',
                           interval_probability_L1_upper=str(2*err),normalization=str(out['Z']),
                           old_intervals_scope='Legacy CDF intervals retained only for old/new basis-law comparison; not used by transformed source.')
    # Keep the raw sampler's reflection and iterate truthful as well as the
    # separately native compile_source output. The reflection is already
    # expanded because it includes synthesized source gates.
    ordinary=out['source_ordinary_bits'];seeds=out['seeds'];target=out['target'];outer=out['outer']
    reflect=inverse(prep)+[('h',q) for q in ordinary]+[('x',q) for q in seeds]
    reflect += [('x',target),('h',target)]+kg_mcx([outer,*seeds],target,qc.scratch[0],True)+[('h',target),('x',target)]
    reflect += [('x',q) for q in reversed(seeds)]+[('h',q) for q in reversed(ordinary)]+prep+[('z',outer)]
    out['reflect']=reflect;out['iterate']=out['B']+out['phase']+out['B'][::-1]+reflect
    out['source_selector_receipt']=dict(old_macro_gates=prefix_end,new_macro_gates=len(source_selector),
        old_CCX=sum(o[0]=='ccx' for o in oldB[:prefix_end]),new_CCX=sum(o[0]=='ccx' for o in source_selector),
        old_selector_raw_bound=str(old_selector_bias),new_source_raw_bound=str(2*out['Z']*err),
        shell_count=J,source_shell_bits=len(ordinal),source_side_bits=1,
        scope='Exact synthesized source and paired inverse; source approximation charged once as a represented-score bias.')
    out['old_source_selector_prefix_end']=prefix_end
    return out


def bridge(b):
    return dict(B=b['B'],controlled_phase=b['B']+b['phase']+b['B'][::-1],registers=b['qc'].regs,
                seedbits=b['seeds'],width=b['qc'].width,outer_control=b['outer'],phase_target=b['target'])


def compile_source(b):
    """Compile compute normally, replace only uniform-seed reflection source."""
    compiled=compile_bundle(bridge(b))
    mapping={int(k):v for k,v in compiled['original_to_compiled_wire'].items()}
    prep=remap_word(b['source_preparation'],mapping)
    ordinary=[mapping[q] for q in b['source_ordinary_bits']]
    seeds=compiled['seedbits'];outer=compiled['outer_control'];target=compiled['phase_target'];scratch=compiled['reflection_scratch']
    reflection=inverse(prep)+[('h',q) for q in ordinary]+[('x',q) for q in seeds]
    reflection += [('x',target),('h',target)]+kg_mcx([outer,*seeds],target,scratch,True)+[('h',target),('x',target)]
    reflection += [('x',q) for q in reversed(seeds)]+[('h',q) for q in reversed(ordinary)]+prep+[('z',outer)]
    compiled['controlled_reflection']=reflection
    compiled['controlled_iterate']=compiled['controlled_phase']+reflection
    compiled['source_preparation']=prep
    compiled['initial_preparation']=[('h',q) for q in ordinary]+prep
    compiled['compiled_ledger']=ledger(compiled['controlled_iterate'],compiled['width'])
    compiled['compiled_ledger']['Toffoli_scope']='Already-expanded native word: accepted relative-phase compute/true adjoint, exact middle and source-preparation gates, coherent KG source reflection.'
    compiled['component_ledgers']['controlled_reflection']=ledger(reflection,compiled['width'])
    compiled['source_preparation_T']=ledger(prep,compiled['width'])['T_count']
    compiled['source_preparation_ledger']=ledger(prep,compiled['width'])
    compiled['initial_preparation_ledger']=ledger(compiled['initial_preparation'],compiled['width'])
    compiled['source_receipt']=b['source_receipt']
    return compiled


def check_source(b,d,old):
    rng=random.Random(746113);regs=b['qc'].regs;ks=d['q_tail']['shell_exponents'];J=len(ks)
    q=lambda bs,bits:sum(((bs>>wire)&1)<<j for j,wire in enumerate(bits))
    witnessed=0
    for s in range(J):
        for side in range(2):
            basis=sum(rng.randrange(2)<<wire for wire in b['source_ordinary_bits'] if wire!=regs['equal_side_source'][0])
            basis|=sum(((s>>i)&1)<<wire for i,wire in enumerate(regs['equal_shell_source']))
            basis|=side<<regs['equal_side_source'][0]
            end,_=classical_apply(b['B'],basis)
            reverse,_=classical_apply(b['B'][::-1],end);assert reverse==basis
            idx=2*s+side;lower=old['intervals'][idx][0]
            oldbasis=sum(((basis>>wire)&1)<<wire for wire in old['seeds'] if wire not in set(regs['seed_select']))
            oldbasis|=sum(((lower>>i)&1)<<wire for i,wire in enumerate(regs['seed_select']))
            oldend,_=classical_apply(old['B'],oldbasis)
            assert q(end,b['addr'])==q(oldend,old['addr'])
            assert ((end>>b['sign'])&1)==((oldend>>old['sign'])&1)
            assert ((end>>b['accept'])&1)==((oldend>>old['accept'])&1)
            witnessed+=1
    # At most 6 live source bits, plus a scratch cleaned by source preparation.
    state=sparse_apply(b['source_preparation'],{0:1},None)
    probs=[0.]*(1<<len(regs['equal_shell_source']))
    for basis,amp in state.items():
        assert basis&~sum(1<<bit for bit in regs['equal_shell_source'])==0
        probs[q(basis,regs['equal_shell_source'])]+=abs(amp)**2
    l1=sum(abs(prob-(1/J if j<J else 0)) for j,prob in enumerate(probs))
    assert l1<2*float(b['source_error'])+1e-12
    rev=sparse_apply(inverse(b['source_preparation']),state,None)
    inverse_garbage=sum(abs(v)**2 for k,v in rev.items() if k)
    assert abs(rev.get(0,0)-1)<1e-12 and inverse_garbage<1e-20
    return dict(valid_shell_side_basis_cases=witnessed,inverse_passed=True,
                fixed_source_probability_L1_observed=l1,
                certified_probability_L1_upper=str(2*b['source_error']),
                invalid_shell_probability=sum(probs[J:]),source_inverse_zero_amplitude_error=abs(rev.get(0,0)-1),
                source_inverse_garbage_norm_squared=inverse_garbage)


def check_source_reflection(b):
    """Native small-space check of both outer branches and absolute phase.

    Every non-shell uniform seed factor commutes with the preparation and is
    treated by the accepted uniform reflection proof. One such seed is kept
    here to test the tensor source; both outer branches have arbitrary input
    amplitudes, so the test is stronger than checking only the source state.
    """
    shell=b['source_shell_bits'];m=len(shell)
    mapping={q:i for i,q in enumerate(shell)}
    oldscratch=b['qc'].scratch[0]
    scratch_bits=list(dict.fromkeys([oldscratch]+b['source_scratch_bits']))
    for i,q in enumerate(scratch_bits):mapping[q]=m+i
    scratch=m;ordinary=m+len(scratch_bits);target=ordinary+1;outer=ordinary+2
    prep=remap_word(b['source_preparation'],mapping)
    seeds=list(range(m))+[ordinary]
    reflection=inverse(prep)+[('h',ordinary)]+[('x',q) for q in seeds]
    reflection += [('x',target),('h',target)]+kg_mcx([outer,*seeds],target,scratch,True)+[('h',target),('x',target)]
    reflection += [('x',q) for q in reversed(seeds)]+[('h',ordinary)]+prep+[('z',outer)]
    psi=sparse_apply([('h',ordinary)]+prep,{0:1},None)
    rng=random.Random(985619)
    state={sum(((j>>i)&1)<<q for i,q in enumerate(seeds))|(c<<outer):complex(rng.random()-.5,rng.random()-.5)
           for j in range(1<<len(seeds)) for c in (0,1)}
    norm=math.sqrt(sum(abs(v)**2 for v in state.values()));state={k:v/norm for k,v in state.items()}
    overlap=sum(v.conjugate()*state.get(k|(1<<outer),0) for k,v in psi.items())
    want={}
    for basis,amp in state.items():
        want[basis]=amp if not ((basis>>outer)&1) else 2*psi.get(basis^(1<<outer),0)*overlap-amp
    actual=sparse_apply(reflection,state,None)
    error=max(abs(actual.get(k,0)-want.get(k,0)) for k in set(actual)|set(want))
    assert error<1e-11,error
    return dict(source_qubits=m,ordinary_uniform_qubits=1,outer_controls=[0,1],
                active_branch='2|implemented_source><implemented_source|-I',
                disabled_branch='identity including absolute phase',
                native_max_amplitude_error=error,support_dimension=len(state))


def main():
    reports=[]
    for n,R in ((12,128),(68,2048)):
        data=ml.fixed_data(1<<n,{'tau':'1','lam':'.1','eta':'.375'})
        raw,old=api.build_from_data(data,R)
        b=transform(old,raw['descriptor']);compiled=compile_source(b)
        bias=data['approximation_bias']+b['bias']
        sched=api.bc.choose_schedule(b['Z'],bias,compiled['compiled_ledger']['T_count'])
        sched['T_count']=sched.pop('T7')+sched['k']*compiled['source_preparation_T']
        row=dict(n=n,R=R,configuration=b['config'],bias=float(bias),schedule=sched,
                 total_label_queries=2*R+sched['quantum_queries'],
                 width_with_phase=compiled['width']+sched['P'].bit_length()-1,
                 source_seed_count=len(b['seeds']),compiled_counts=compiled['compiled_ledger'],
                 initial_source_ledger=compiled['initial_preparation_ledger'],
                 selector=b['source_selector_receipt'],source=b['source_receipt'],
                 verification=check_source(b,raw['descriptor'],old),reflection_verification=check_source_reflection(b),
                 scope='Complete accepted linear response with replaced equal-shell source, no address/response optimizations. Actual synthesized source words certified projectively and paired with their true adjoints. Initial source preparation included.')
        reports.append(row)
        path=ROOT/f'results/response/autonomous_blind_source_n{n}.json'
        path.write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(row,indent=2),flush=True)
    (ROOT/'results/response/autonomous_blind_source_summary.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')


def generic_selftest():
    rows=[]
    for J in (1,2,3,4,5,6,7,9,10,11,13,17,31,56):
        m=(J-1).bit_length();bits=list(range(m));scratch=[m,m+1]
        word,error,receipt=source_word(bits,scratch,J)
        state=sparse_apply(word,{0:1},None)
        probabilities=[0.]*(1<<m)
        for basis,amp in state.items():
            assert basis>>m==0, (J,basis)
            probabilities[basis]=abs(amp)**2
        l1=sum(abs(v-(1/J if i<J else 0)) for i,v in enumerate(probabilities))
        assert l1<2*float(error)+1e-12,(J,l1,float(error))
        back=sparse_apply(inverse(word),state,None)
        garbage=sum(abs(v)**2 for k,v in back.items() if k)
        assert abs(back.get(0,0)-1)<1e-12 and garbage<1e-20
        rows.append(dict(J=J,T=sum(o[0] in ('t','tdg') for o in word),
                         L1_observed=l1,L1_upper=str(2*error),inverse_garbage_norm_squared=garbage))
    path=ROOT/'results/response/autonomous_blind_source_generic.json'
    path.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    if '--selftest-generic' in sys.argv:generic_selftest()
    else:main()
