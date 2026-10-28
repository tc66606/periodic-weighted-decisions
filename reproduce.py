"""Portable verification. Input files are immutable; reports use a fresh work copy.

python -B reproduce.py --mode math --output run-math
python -B reproduce.py --mode full --output run-full
The full mode includes all 80 native resource rows, bounded finite menus, source
receipts, and conditional mathematical certificates. It executes verification,
not the exponentially large quantum programs on hardware or a simulator.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, csv, ctypes, gzip, hashlib, importlib.util, json, os
import runpy, shutil, subprocess, sys, time

RELEASE=Path(__file__).resolve().parent
NAME='heat_weight_randomized_sign_frontier'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p); b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def contained(p,root):return p==root or p.is_relative_to(root)

def install_guard(work):
    """Deny file reads outside this extracted work copy and Python runtime.

    This is an auditable application guard, not an OS security sandbox. It
    intentionally gives no access to original scientific folders. All writes
    must remain in the work copy. Imported third-party libraries are installed,
    never copied from an old scientific tree or discovered through PYTHONPATH.
    """
    allowed=[work.resolve(),Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
    reads=set()
    def audit(event,args):
        if event=='socket.connect':raise PermissionError('Network is forbidden in replay')
        if event not in ('open','os.listdir','os.scandir'):return
        value=args[0]
        if value is None or isinstance(value,int):return
        p=Path(os.fsdecode(value)).resolve()
        writing=False
        if event=='open':
            mode=args[1]; flags=args[2]
            writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:
            if not contained(p,work):raise PermissionError('Replay write outside disposable work copy: '+str(p))
        elif not any(contained(p,a) for a in allowed):
            raise PermissionError('Replay scientific read outside extracted inputs/runtime: '+str(p))
        if event=='open' and not writing and contained(p,work):reads.add(p.relative_to(work).as_posix())
    sys.addaudithook(audit)
    return reads

def check_bindings():
    if (RELEASE/'manifest.json').exists():
        for entry in read(RELEASE/'manifest.json')['files']:
            p=(RELEASE/entry['artifact_id']).resolve()
            assert contained(p,RELEASE) and sha(p)==entry['sha256'],entry['artifact_id']
    provenance=read(RELEASE/'provenance.json')['files']
    for item in provenance:
        p=RELEASE/item['artifact_id']
        assert p.is_file() and sha(p)==item['released_sha256'],item['artifact_id']
    with (RELEASE/'resources.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==80
    seen=set()
    for r in rows:
        key=tuple(r[k] for k in ('tag','response','route','S')); assert key not in seen;seen.add(key)
        for label in ('program','certificate','raw','native'):
            p=(RELEASE/r[label+'_artifact_id']).resolve()
            assert contained(p,RELEASE) and sha(p)==r[label+'_sha256']
        program=read(RELEASE/r['program_artifact_id']); raw=read(RELEASE/r['raw_artifact_id'])
        e=program['error']; a=program['resources']; ob=program['observable'];d=raw['descriptor']
        for k in ('P','S','K'):assert int(r[k])==program[k]
        for k in ('Z','bias'):assert F(r[k])==F(ob[k])
        for k in ('n','tau','lambda','eta'):assert str(r[k])==str(d[k])
        assert int(r['head_calls'])==ob['head_labels']==a['head_queries']
        assert r['response_name']==('cubic' if r['response']=='nodes' else 'linear')
        assert r['phase_status']==r['submission_status']=='prior_used'
        assert F(r['failure'])==F(e['total'])<F(1,20)
        for name,value in [('queries',a['total_queries']),('longest_chain',a['longest_chain']),('T_worst',a['worst']['T']),
                           ('T_depth_worst',a['worst']['T_depth']),('nonoracle_depth_worst',a['worst']['depth']),('width',a['allocated_width'])]:
            assert int(r[name])==value,(key,name)
    return dict(status='passed',files=len(provenance),rows=80,scope='Byte integrity, parameters and exact public table/program bindings')

def retained_finite_query(root):
    tight=root.parent/'heat_weight_tight_frontier'
    data=read(tight/'results/next/finite_query_certificates.json')
    candidates=data['controls']+data['discovery']+[data['best']]
    row=next(r for r in candidates if r['n']==68 and r['R']==8111 and r['rho']=='.02')
    check=module('retained_independent_theory',tight/'independent_checks/next/fresh_theory_check.py')
    result=check.independent_evaluate(row)
    result['claim_scope']='Retained finite numerical corollary conditional on the analytic lower-bound theorem; not automatic proof verification or a compiled quantum upper bound.'
    return result

def math_check(root):
    cv=module('independent_grid',root/'independent_checks/random_sign/review_certificates.py');cv.side_conditions()
    unique={}
    for p in sorted((root/'results/random_sign/programs').glob('*.json')):
        program=read(p);ob=program['observable'];cert=Path(program['certificate']['path']).resolve()
        unique[str(cert)]=(ob['Z'],ob['bias'])
    checked=[cv.review(Path(p),expected=values) for p,values in unique.items()]
    result=dict(status='passed',programs=80,unique_certificates=len(checked),certificates=checked,
      retained_finite_query=retained_finite_query(root),
      claim_scope='Conditional finite-grid mathematics for declared Z/beta, and retained finite-query numerical corollary. Does not certify source values/native words/full program error by itself.')
    (root/'results/random_sign/release_math.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def discovery_sources(root):
    r=module('retained_source_checker',root.parent/'heat_weight_response_frontier/independent_checks/response/review_final.py')
    coord=root.parent/'heat_weight_coordinate_frontier/results/coordinate'; rows=[]
    for tag,layout in [('discovery_n12','tapered'),('discovery_n68','parallel')]:
        for method in ('linear','nodes'):
            rawpath=coord/f'{tag}_{method}_{layout}_raw.json.gz';wordpath=coord/f'{tag}_{method}_{layout}_word.json.gz'
            raw,word=r.read(rawpath),r.read(wordpath)
            eps,source=r.source_replay(raw);bias,tables=r.tables_and_bias(raw,eps)
            descriptor=r.descriptor_check(raw);native=r.native_word_check(raw,word)
            assert bias==F(raw['raw_bias_fraction'])<F(1,5000)
            rows.append(dict(tag=tag,response=method,raw_sha256=sha(rawpath),native_sha256=sha(wordpath),
                 bias=str(bias),source=source,tables=tables,descriptor=descriptor,native=native))
            print('source passed',tag,method,flush=True)
    result=dict(status='passed',cases=len(rows),rows=rows,
      claim_scope='Retained discovery source receipts, response tables, descriptor and literal native word reconstruction. Generic coordinate-construction theorem retained; no old scan rerun.')
    (root/'results/random_sign/release_discovery_sources.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def worker(work,step):
    root=work/'scientific'/NAME;os.chdir(root)
    sys.path.insert(0,str(root/'independent_checks/random_sign'))
    reads=install_guard(work)
    # A real file outside the work copy must be unreadable after the guard.
    try:(RELEASE/'reproduce.py').read_bytes()
    except PermissionError:pass
    else:raise AssertionError('Outside-root read guard did not reject the probe')
    if step=='math':result=math_check(root)
    elif step=='discovery_sources':result=discovery_sources(root)
    else:
        script, *args=step.split(':');p=root/'independent_checks/random_sign'/f'review_{script}.py'
        sys.argv=[str(p),*args];runpy.run_path(str(p),run_name='__main__');result={'status':'passed'}
    (work/f'memory_{step.replace(":","_")}.json').write_text(json.dumps(dict(
      peak_process_working_set_bytes=peak_memory(),method='GetProcessMemoryInfo on this verification process, not the virtual-environment launcher'))+'\n')
    (work/f'access_{step.replace(":","_")}.json').write_text(json.dumps(dict(status='passed',outside_root_probe='rejected',
      guard='Reads limited to extracted scientific inputs and the installed Python runtime; writes limited to disposable work copy.',
      scientific_reads=sorted(reads)),indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('status','cases','programs','unique_certificates')}),flush=True)

def peak_memory():
    if os.name!='nt':return None
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in (
          'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
          'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    pm=PMC();pm.cb=ctypes.sizeof(pm)
    fn=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
    fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD];fn.restype=wintypes.BOOL
    # Query inside the worker. Windows virtual-environment python.exe may be a
    # small redirector process; measuring its parent Popen handle undercounts.
    if not fn(wintypes.HANDLE(-1),ctypes.byref(pm),pm.cb):return None
    return pm.PeakWorkingSetSize

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('bindings','math','full'),default='math')
    ap.add_argument('--output',type=Path,default=Path('replay-output'))
    ap.add_argument('--worker');ap.add_argument('--work-root',type=Path)
    args=ap.parse_args()
    if args.worker:worker(args.work_root.resolve(),args.worker);return
    started=time.perf_counter();bindings=check_bindings();out=args.output.resolve()
    assert not out.exists(),'Choose a new output directory; existing evidence is never overwritten.'
    assert not contained(out,RELEASE/'scientific'),'Output must not be inside scientific inputs.'
    out.mkdir(parents=True);work=out/'work';shutil.copytree(RELEASE/'scientific',work/'scientific')
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    steps=[] if args.mode=='bindings' else ['math']
    if args.mode=='full':steps+=['ramps','native_programs','math:identity','certificates:--menu','sources','discovery_sources','program_ledgers','final']
    measurements=[]
    for step in steps:
        actual=step
        log=out/(actual.replace(':','_')+'.log');begin=time.perf_counter()
        command=[sys.executable,'-I','-B',str(RELEASE/'reproduce.py'),'--worker',actual,'--work-root',str(work)]
        print('VERIFY',actual,flush=True)
        with log.open('w',encoding='utf8') as stream:
            p=subprocess.Popen(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT);code=p.wait()
        memoryfile=work/f'memory_{actual.replace(":","_")}.json'
        peak=read(memoryfile)['peak_process_working_set_bytes'] if memoryfile.exists() else None
        row=dict(step=actual,wall_seconds=time.perf_counter()-begin,peak_process_working_set_bytes=peak,exit_code=code,log=log.name)
        measurements.append(row)
        if code:
            print(log.read_text(encoding='utf8'));raise SystemExit(code)
    # Recheck input immutability after all verification steps.
    assert check_bindings()==bindings
    import importlib.metadata
    required=['mpmath'] if args.mode=='math' else ['mpmath','numpy','scipy'] if args.mode=='full' else []
    summary=dict(status='passed',mode=args.mode,python=sys.version.split()[0],dependencies={p:importlib.metadata.version(p) for p in required},
      verified_payload_sha256={name:sha(RELEASE/name) for name in ('reproduce.py','provenance.json','resources.csv','controls.csv','requirements-math.txt','requirements-full.txt')},
      bindings=bindings,steps=measurements,wall_seconds=time.perf_counter()-started,
      peak_worker_working_set_bytes=max((r['peak_process_working_set_bytes'] or 0 for r in measurements),default=0) or None,
      memory_method='Windows GetProcessMemoryInfo PeakWorkingSetSize per verification process; suite reports max worker, excludes filesystem cache/parent. Null on unsupported platforms.',
      input_files_unchanged=True,original_scientific_roots_access='forbidden by per-worker allowlist',
      limitations=['Installed dependencies and Python runtime are external prerequisites; this run does not prove a fresh pip installation.',
       'No hardware execution. No generic coordinate theorem or analytic lower-bound theorem is proved automatically.',
       'Math mode is conditional on declared observable normalization/bias and is not the complete native gate certificate.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
