"""Affected small coherent interfaces; the continuum guarantee is a separate proof."""
from pathlib import Path
from fractions import Fraction as F
import importlib.util, json, math, sys, unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/random_sign'))
import native

def simulate(word,state):
    a=np.asarray(state,dtype=complex).copy()
    matrices={'h':np.array([[1,1],[1,-1]])/np.sqrt(2),
      'x':np.array([[0,1],[1,0]]),'z':np.diag([1,-1]),
      's':np.diag([1,1j]),'sdg':np.diag([1,-1j]),
      't':np.diag([1,np.exp(1j*np.pi/4)]),'tdg':np.diag([1,np.exp(-1j*np.pi/4)])}
    for op in word:
        if op[0]=='cx':
            c,t=op[1:]
            for i in range(len(a)):
                if (i>>c)&1 and not (i>>t)&1:
                    j=i^(1<<t);a[i],a[j]=a[j],a[i]
        else:
            q=op[1];g=matrices[op[0]]
            for lo in range(len(a)):
                if not (lo>>q)&1:
                    hi=lo^(1<<q);a[lo],a[hi]=g@np.array([a[lo],a[hi]])
    return a

class NativeDecision(unittest.TestCase):
    observations=[]
    def test_random_and_padded_coherent_interfaces(self):
        worst=0.;inverse_error=0.;cases=0
        for m in (3,4,5):
            P=1<<m;t=np.arange(P);cw,_=native.clock(m)
            initial=np.zeros(P,complex);initial[0]=1
            source=simulate(cw['word'],initial)
            ideal=np.sqrt(2/P)*np.sin(np.pi*(t+.5)/P)
            for S in (1,2,4):
                ramp,_=native.ramps(m,S);qft,_=native.qft(m);padft,_=native.qft(m+S.bit_length()-1)
                for phi in (0.,np.pi,2*np.pi-1e-9,.371,-.371,np.pi/P,-np.pi/P):
                    aggregate=np.zeros(S*P)
                    ideal_joint=np.zeros(S*P)
                    for b in ramp['branches']:
                        z=b['z'];a=simulate(b['word'],source)
                        restored=simulate(b['inverse_word'],a)
                        inverse_error=max(inverse_error,float(np.linalg.norm(restored-source)))
                        a*=np.exp(1j*t*phi)
                        actual=simulate(qft['word'],a)
                        expected=np.fft.fft(ideal*np.exp(1j*t*(phi-2*np.pi*z/(S*P))))/np.sqrt(P)
                        err=float(np.sum(np.abs(np.abs(actual)**2-np.abs(expected)**2))/2)
                        allowance=2*float(F(cw['operator_error'])+F(b['operator_error'])+F(qft['operator_error']))+2e-12
                        self.assertLessEqual(err,allowance)
                        aggregate[z::S]=np.abs(actual)**2/S
                        ideal_joint[z::S]=np.abs(expected)**2/S
                        worst=max(worst,err);cases+=1
                    padded=np.zeros(S*P,complex);padded[:P]=source*np.exp(1j*t*phi)
                    self.assertEqual(float(np.linalg.norm(padded[P:])),0.)
                    actual_pad=np.abs(simulate(padft['word'],padded))**2
                    ideal_pad=np.abs(np.fft.fft(np.pad(ideal*np.exp(1j*t*phi),(0,(S-1)*P)))/np.sqrt(S*P))**2
                    self.assertLess(float(np.max(np.abs(ideal_joint-ideal_pad))),3e-14)
                    allowance=2*float(F(cw['operator_error'])+F(padft['operator_error']))+2e-12
                    self.assertLessEqual(float(np.sum(np.abs(actual_pad-ideal_joint))/2),allowance)
                    self.assertLess(abs(float(sum(aggregate))-1),2e-11)
        self.assertLess(inverse_error,2e-11)
        self.observations.append(dict(native_branch_cases=cases,worst_TV=worst,literal_inverse_error=inverse_error))

    def test_decoder_boundaries_and_ties(self):
        for S in (1,2,4,8,16):
            for j in (0,1,7,8,15,16,31):
                for z in range(S):
                    value=native.signed_mean(j,32,z,S)
                    target=math.cos(2*math.pi*(j+z/S)/32)
                    self.assertLess(abs(float(value)-target),float(F(1,2**48))+1e-15)
        self.assertEqual(native.decision([(0,0)],32,4,F(1),F(-1)),1)
        self.assertEqual(native.decision([(16,0)],32,4,F(1),F(1)),1)
        with self.assertRaises(ValueError):native.decision([(0,0),(1,0)],32,4,F(1))
        with self.assertRaises(ValueError):native.signed_mean(0,32,4,4)

    def test_reusing_offset_is_not_independent_amplification(self):
        majority=lambda p:3*p*p-2*p*p*p
        p=F(1,5)
        fresh=majority(p);shared=(majority(F(0))+majority(F(2,5)))/2
        self.assertEqual(fresh,F(13,125));self.assertEqual(shared,F(22,125))
        self.assertGreater(shared,fresh)

    def test_ramp_sign_negative_control(self):
        P,S,z=32,4,3;t=np.arange(P);phi=.371
        a=np.sqrt(2/P)*np.sin(np.pi*(t+.5)/P)
        negative=np.abs(np.fft.fft(a*np.exp(1j*t*(phi-2*np.pi*z/(S*P))))/np.sqrt(P))**2
        wrong=np.abs(np.fft.fft(a*np.exp(1j*t*(phi+2*np.pi*z/(S*P))))/np.sqrt(P))**2
        self.assertGreater(float(np.sum(np.abs(negative-wrong))/2),.5)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(NativeDecision)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=dict(passed=result.wasSuccessful(),tests=result.testsRun,observations=NativeDecision.observations,
      scope='Actual native small clock/ramp/IQFT and adjoints with eigenphase controlled-power interface. Exact analytical POVM comparison. No full many-qubit heat computation; interval proofs are separate.')
    (ROOT/'results/random_sign/targeted_semantics.json').write_text(json.dumps(out,indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
