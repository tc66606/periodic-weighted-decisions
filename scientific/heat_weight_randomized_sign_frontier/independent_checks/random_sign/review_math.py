"""Independent adversarial convention checks for finite random clock offsets.

No constructor imports. Floating calculations only falsify finite identities;
they never certify the continuum tail probability used by a schedule.
"""
from pathlib import Path
from fractions import Fraction
import json
import math
import numpy as np
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[2]


def lattice_tail(f, d, b):
    # Inclusive lower bounds. Only call away from a floating event switch.
    plus = math.ceil(f + d) - f
    minus = math.ceil(b - f) + f
    return math.sin(math.pi*f)**2 / math.pi**2 * (polygamma(1, plus) + polygamma(1, minus))


def experiment(P, S, phi, gap, kind='uniform'):
    t = np.arange(P)
    amplitude = np.ones(P)/math.sqrt(P) if kind=='uniform' else np.sqrt(2/P)*np.sin(math.pi*(t+.5)/P)
    joint = []
    for z in range(S):
        delta = z / S
        state = np.exp(1j * (phi - 2*math.pi*delta/P) * t) * amplitude
        weights = abs(np.fft.fft(state) / math.sqrt(P))**2 / S
        decoded = np.cos(2*math.pi*(t+delta)/P)
        joint.extend(zip(decoded, weights))
    lower = sum(w for c, w in joint if c <= math.cos(phi)-gap)
    upper = sum(w for c, w in joint if c >= math.cos(phi)+gap)
    return float(lower), float(upper), joint


def sorted_distribution(joint):
    out = {}
    for cosine, weight in joint:
        key = round(float(cosine), 12)
        out[key] = out.get(key, 0) + float(weight)
    return out


def main():
    rows = []
    worst_symmetry = 0.
    worst_margin = 1.
    for P in (4, 8, 16, 32):
        for S in (1, 2, 4, 8):
            for gap in (.03, .2, .8):
                d = P*gap/(2*math.pi)
                b = P*math.sqrt(2*gap)/(2*math.pi)
                for phi in (0., 1e-9, .317, math.pi/2-.07, math.pi/2,
                            math.pi/2+.07, math.pi-.013, math.pi):
                    lo, hi, joint = experiment(P, S, phi, gap)
                    neglo, neghi, negjoint = experiment(P, S, -phi, gap)
                    complo, comphi, compjoint = experiment(P, S, math.pi-phi, gap)
                    sym = max(abs(lo-neglo), abs(hi-neghi), abs(lo-comphi), abs(hi-complo))
                    assert sym < 1e-12, (P,S,phi,sym)
                    worst_symmetry = max(worst_symmetry, sym)
                    f0 = (P*phi/(2*math.pi)) % (1/S)
                    bound = sum(lattice_tail(f0+r/S, d, b) for r in range(S))/S
                    assert lo <= bound + 1e-12, (P,S,phi,gap,lo,bound)
                    worst_margin = min(worst_margin, bound-lo)
                    rows.append(dict(P=P,S=S,phi=phi,gap=gap,lower=lo,upper=hi,lower_tail_bound=bound))
    sine_cases=0
    sine_symmetry=0.
    for P in (4,8,16,32):
        for S in (1,2,4,8):
            for phi in (0., .317, math.pi/2-.07, math.pi/2, math.pi-.013, math.pi):
                for gap in (.03,.2,.8):
                    lo,hi,_=experiment(P,S,phi,gap,'sine')
                    nlo,nhi,_=experiment(P,S,-phi,gap,'sine')
                    clo,chi,_=experiment(P,S,math.pi-phi,gap,'sine')
                    err=max(abs(lo-nlo),abs(hi-nhi),abs(lo-chi),abs(hi-clo))
                    assert err<1e-12
                    sine_symmetry=max(sine_symmetry,err);sine_cases+=1
    # A single fixed offset lacks the negative-eigenphase symmetry. Require
    # the random reindexing in the mathematical proof rather than fixed-z.
    P, S, z, phi, gap = 8, 4, 1, .317, .2
    t = np.arange(P)
    def fixed_bad(sign):
        v = np.exp(1j*(sign*phi-2*math.pi*z/(S*P))*t)/math.sqrt(P)
        p = abs(np.fft.fft(v)/math.sqrt(P))**2
        c = np.cos(2*math.pi*(t+z/S)/P)
        return float(p[c <= math.cos(phi)-gap].sum())
    fixed = [fixed_bad(1), fixed_bad(-1)]
    assert abs(fixed[0]-fixed[1]) > 1e-3
    # Reusing one offset across runs does not justify a binomial at its
    # averaged failure rate: this elementary exact example detects it.
    p = Fraction(1,5)
    independent_majority = 3*p*p*(1-p)+p**3
    reused_majority = Fraction(1,2)*(3*Fraction(2,5)**2*Fraction(3,5)+Fraction(2,5)**3)
    assert reused_majority > independent_majority
    # Phase-model controls at the actual target certificate's exact gap.
    # They falsify dropping the randomization or reusing a sampled branch;
    # they make no claim that this phase is realized by a particular label input.
    cert_path=ROOT/'results/random_sign/bound_discovery_n68_nodes_P16384_S4.json'
    cert=json.loads(cert_path.read_text())['certificate']
    Ptarget,Starget=cert['P'],cert['S'];g=float(Fraction(cert['g']))
    phase=math.acos(g+1e-9);t=np.arange(Ptarget)
    a=np.sqrt(2/Ptarget)*np.sin(math.pi*(t+.5)/Ptarget)
    conditional=[]
    for z in range(Starget):
        state=a*np.exp(1j*t*(phase-2*math.pi*z/(Ptarget*Starget)))
        mass=abs(np.fft.fft(state)/math.sqrt(Ptarget))**2
        decoded=np.cos(2*math.pi*(t+z/Starget)/Ptarget)
        conditional.append(float(mass[decoded<=math.cos(phase)-g].sum()))
    average=sum(conditional)/Starget
    independent_actual=3*average**2-2*average**3
    shared_actual=sum(3*p**2-2*p**3 for p in conditional)/Starget
    assert conditional[0]>.04 and average<float(Fraction(cert['statistical_failure']))
    assert shared_actual>independent_actual
    actual_control=dict(P=Ptarget,S=Starget,phase=phase,conditional_failures=conditional,
                        averaged_failure=average,independent_K3=independent_actual,shared_K3=shared_actual,
                        scope='Numerical phase-model counterexample; not a new all-label lower bound.')
    result = dict(status='passed',cases=len(rows),maximum_joint_symmetry_error=worst_symmetry,
                  sine_clock_cases=sine_cases,sine_clock_maximum_joint_symmetry_error=sine_symmetry,
                  minimum_tail_slack=worst_margin,fixed_offset_counterexample=dict(P=P,S=S,z=1,phi=phi,gap=gap,failures=fixed),
                  shared_offset_counterexample=dict(independent=str(independent_majority),reused=str(reused_majority)),
                  target_sine_controls=actual_control,
                  scope='Small finite Fourier convention tests and adversarial examples only. No continuum probability or candidate schedule is certified.',
                  rows=rows)
    out=ROOT/'results/random_sign/review_math.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
