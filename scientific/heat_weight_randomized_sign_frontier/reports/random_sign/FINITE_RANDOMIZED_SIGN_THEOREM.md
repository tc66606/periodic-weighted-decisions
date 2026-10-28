# Finite random-offset sine-clock sign theorem

This is an analytic certificate for the actual finite classical offset family.
The guaranteed quantity is

\[
\sup_{\phi}\frac1S\sum_{z=0}^{S-1}
 \Pr[\text{wrong sign}\mid\phi,z].
\]

There is one quantum run, with one uniform classical choice of `z`.  The sum
describes alternatives, not `S` executed programs.  The proof never replaces
this quantity by an average over unknown input phases.  It also retains the
averaging operation before taking the maximum of the interval bounds.

## Fixed construction and scope

Let `P=2^m`, `m=1,...,18`, `S in {1,2,4,8,16}`, and let an optional repetition
count `K` be odd in `1,...,127`.  Every accepted source/method pair uses these
same menus and its own exact saved `Z` and `beta`.  No rounded bias is used.
The clock state is

\[
 a_t=\sqrt{2/P}\sin\bigl(\pi(t+1/2)/P\bigr),\qquad 0\le t<P.
\]

Draw `z` uniformly from `0,...,S-1`, set `xi=z/S`, and apply the diagonal
clock ramp `exp(-2 pi i xi t/P)` before the existing controlled iterate and
inverse QFT.  The output angle is `2 pi (j+xi)/P`, not `2 pi j/P`.  The same
head contribution, source, iterate, bias, and promised raw margin remain in
force.  The finite classical cosine decoder has error at most `2^-48` at
these shifted rational angles.

The numerical construction is fixed at 128 equal closed base cells in
`[0,1/S]`, eight explicit terms on each ray, the near split `P/16`, and
`a=127/128`.  It uses exact rational arithmetic throughout.  Square-root
lower bounds use 80 fractional bits; cosine, ray, and probability upper
bounds are rounded upward to 64, 96, and 80 fractional bits respectively.
The pi endpoints are

```
3.14159265358979323846264338327950288419716939937510
3.14159265358979323846264338327950288419716939937511
```

The producer verifies this bracket using Machin's identity and 48 exact
alternating terms for each arctangent.  The bracket and every subsequent
rounding direction are part of the certificate; displayed decimals are
informational.  Small sizes outside the applicability condition `P>=32` and
`q_forward>3/4` receive the valid universal upper bound one, with no claim
that a different analysis could not certify them.

## All-label reduction, including every head transcript

For an exact measured-head sum `c_H`, the implemented score is
`c_H+Z cos(phi)`, and the all-label raw bias is at most `beta`.  Let

\[
 \gamma=1/500,\qquad
 g=(\gamma-\beta)/Z-2^{-48}>0.
\]

On a positive promised input, an incorrect sign, including a tie, implies

\[
 \cos(2\pi(j+\xi)/P)\le\cos\phi-g.
\]

On a negative promised input, it implies the corresponding upper error
`cos(2 pi (j+xi)/P)>=cos(phi)+g`.  This algebra uses no restriction on the
head-dependent threshold `-c_H/Z`; in particular it remains true when that
threshold lies outside `[-1,1]`.  The proof thus covers every reachable head
transcript and every promised label assignment.  The actual source's bias
is already included in `beta`.

## Exact kernel and finite-offset symmetries

For an eigenphase `phi`, write `x=P phi/(2 pi)` and `d=x-j-xi`, reduced modulo
`P` whenever convenient.  The ramp makes the effective ordinary Fourier
position `x-xi`.  The exact measurement law is

\[
 k_P(d)=\frac{2}{P^2}\,
 \frac{\cos^2(\pi d)\cos^2(\pi d/P)\sin^2(\pi/(2P))}
 {[\sin^2(\pi d/P)-\sin^2(\pi/(2P))]^2}.
\]

This follows by expanding the sine clock into two complex exponentials and
evaluating the two finite geometric sums.  The continuous values at
`d=+/-1/2 mod P` are `1/2`, as is also immediate directly from those sums.
The clock is normalized because the sum of its sine-squared samples is
`P/2`.  The kernel is even and `P`-periodic for even `P`.

Each pair `(z,j)` corresponds uniquely to a point `y=j+z/S` on the finite
grid of spacing `1/S`, modulo `P`, with joint probability `k_P(x-y)/S`.
Reflection `y -> -y mod P` permutes that grid, including the offset labels
`z -> -z mod S`; at nonzero `z`, its integer part changes as needed.  Thus
`x -> -x` preserves the averaged distribution of the decoded cosine.  This
proves the same sign bound for both eigenphases of the unchanged iterate,
without any assumption on their spectral weights.

Since `P` is even, `x -> P/2-x` and `y -> P/2-y` likewise permute the finite
grid, preserve the kernel, and negate both cosines.  Hence a uniform lower
cosine-error certificate implies the identical upper-error certificate.
These are symmetries of the actual finite mixture, not an assertion that an
individual nonzero-offset branch is reflection symmetric.  It suffices to
study lower errors with `0<=phi<=pi`.

## Forward and backward circular-ray separation

If `cos(phi)-g<-1`, the lower-error event is empty.  Otherwise define
`alpha=acos(cos(phi)-g)`, so `phi<alpha<=pi`.  The bad output angles in
`[0,2pi]` form the closed interval `[alpha,2pi-alpha]`.  Its real lifts
relative to `phi` are

\[
 [\alpha-\phi+2\pi n,\;2\pi-\alpha-\phi+2\pi n],\quad n\in\mathbb Z.
\]

Every positive member of these intervals is at least `alpha-phi>=g`, by
the unit Lipschitz constant of cosine.  Every negative member has magnitude
at least `alpha+phi`.  Because `cos(alpha)=cos(phi)-g<=1-g` and
`1-cos(alpha)<=alpha^2/2`, we have

\[
 \alpha+\phi\ge\alpha\ge\sqrt{2g}.
\]

Choose the shortest signed circular representative of each outcome error,
whose magnitude is at most `pi`.  It is one of these real lifts, so the same
directional separations apply: forward distance at least `g`, backward
distance at least `sqrt(2g)`.  At antipodal outcomes either orientation may
be chosen consistently; counting both would only enlarge the bound.  This
step justifies the circular distance used in the sine kernel and the
asymmetric thresholds simultaneously.  It does not unfold the sine kernel
into a nonperiodic probability law.

Let `sqrt_lower(2g)` be the exact dyadic lower square root and define

\[
 q_f=Pg/(2\pi_+),\qquad
 q_b=P\,\operatorname{sqrt\_lower}(2g)/(2\pi_+).
\]

Both are certified lower bounds on the corresponding bin distances.  The
square-root routine verifies `r^2<=2g<(r+2^-80)^2` by integer arithmetic.

## Near and far bounds, without singular points

Let `d` denote the positive magnitude of a signed circular bin error.
For `d<=P/16`,

\[
 \sin(\pi d/P)\ge a\pi d/P,\qquad a=127/128.
\]

Indeed `sin(u)/u>=1-u^2/6`, and the producer verifies
`1-pi_+^2/(6*16^2)>a`.  If `C` bounds the common numerator
`cos^2(pi*d)`, and `d>1/(2a)`, the exact kernel obeys

\[
 k_P(d)\le\frac{C}{2\pi_-^2}\,F(d),\qquad
 F(d)=\frac1{(a^2d^2-1/4)^2}.
\]

Here both sine bounds initially use the same exact pi: the denominator is
at least `(pi/P)^2(a^2 d^2-1/4)`, while the numerator is at most
`C*pi^2/(2P^4)`.  Only then is the remaining `1/pi^2` replaced by
`1/pi_-^2`.  The function `F` is positive and decreasing on this domain.

On the far region `P/16<d<=P/2`, sine is monotone up to `pi/2`, so
`sin(pi*d/P)>=sin(pi/16)>3/16`; the latter inequality follows from
`pi_-/16-(pi_+/16)^3/6>3/16`.  Also `sin(pi/(2P))<2/P`.
There are at most `P` outcomes per branch, so their entire far contribution
is bounded by

\[
 B_{\rm far}=\frac{8}{P^3(9/256-4/P^2)^2}.
\]

Its denominator is positive for `P>=32`.  This same bound holds after the
average over branches; there is no factor `S`.  Near and far are disjoint at
`P/16`, though including their common boundary twice would be conservative.
The removable kernel points at `d=1/2` never enter this proof's bad near
tails: `q_f>3/4`, and for the accepted inputs `q_b>q_f`.  Smaller parameters
use the universal cap instead of attempting division at a singularity.

For a ray starting at any `v>1/(2a)`, define

\[
 R(v)=\sum_{j=0}^7 F(v+j)
 +\frac1{3(v+7)^3[a^2-1/(4(v+7)^2)]^2}.
\]

The omitted terms have `j>=8`.  Since `F` decreases, their sum is at most
the integral of `F(t)` from `v+7` to infinity.  On that interval,
`F(t)<=t^-4[a^2-1/(4(v+7)^2)]^-2`, which integrates to the displayed
remainder.  Extending a finite circular near ray to this positive infinite
series only increases its bound, including when the near ray is empty.

## Closed cells and averaging before supremum

Define `f_z=frac(z/S-x)`.  A forward lattice distance has the form `f_z+n`,
and a backward one has the form `n-f_z`, for integer `n`.  The numerator of
every such outcome is `cos^2(pi*f_z)`.  This sign convention matters: it
explains which of the two asymmetric thresholds belongs to each formula.

For a closed fractional cell `[L,U]` and any contained `f_z`, all forward
outcomes with distance at least `q_f` have integer index at least
`ceil(q_f-U)`.  All backward outcomes at distance at least `q_b` have index
at least `ceil(q_b+L)`.  Therefore lower ray arguments valid throughout the
closed cell are

\[
 v_f=\lceil q_f-U\rceil+L,\qquad
 v_b=\lceil q_b+L\rceil-U.
\]

These statements use non-strict inequalities and hold exactly at every
event switch and cell boundary, including when a ceiling argument is an
integer.  They do not assume that the same outcomes remain bad throughout
the cell.  Since each cell width is at most `1/128`, both arguments exceed
`3/4-1/128>1/(2a)` for the accepted nontrivial cases.  The endpoint values
`f=0` and `f=1` enumerate the same integer lattice, with the integer index
shifted; including either or both is valid.

Let `C(L,U)` be a rational upper bound on `max cos^2(pi*f)` in that closed
cell.  The producer obtains it without an interval library or numerical
sampling.  Put `h=max(|L-1/2|,|U-1/2|)`.  Then the maximum is
`sin^2(pi*h)`.  Use one for `h=1/2`; otherwise the alternating sine Taylor
polynomial through degree 17 is an upper bound.  Its positive monomials
use `pi_+*h`, and its negative monomials use `pi_-*h`.  This polynomial is
nonnegative, so its square bounds the squared sine.  Round upward to 64
bits and cap at one.  The alternating remainder is valid because all its
term magnitudes decrease on `[0,pi/2]`.

An outward near bound for one translated cell is consequently

\[
 H(L,U)=\frac{C(L,U)}{2\pi_-^2}\,[R(v_f)+R(v_b)].
\]

As a multiset, `{frac(f+z/S):0<=z<S}` is invariant under `f -> f+1/S`.
Thus the average of these periodic ray envelopes has period `1/S`, and
its supremum is attained within the covered base interval `[0,1/S]`.
Partition this base interval into 128 closed cells
`I_i=[i/(128S),(i+1)/(128S)]`.  On each base cell translate both endpoints
by `z/S`, for each `z`.  All translated cells lie in `[0,1]`; the final
endpoint one is handled as above.  For any fixed phase inside `I_i`,

\[
 \frac1S\sum_z p_{{\rm bad,near},z}(\phi)
 \le\frac1S\sum_z H(I_i+z/S).
\]

The certified uniform single-run statistical failure is

\[
 p_{P,S}=\min\left\{1,\;
 B_{\rm far}+\max_{0\le i<128}\frac1S\sum_{z=0}^{S-1}
 H(I_i+z/S)\right\}.
\]

Every arithmetic quantity is rounded only outward, as documented above.
The producer saves all translated cells, their ray arguments, cosine
bounds, and averaged bounds for the target certificates.  This is an
interval cover of a continuum.  A phase grid, fitted curve, or observed
maximum supplies no part of the guarantee.

## Native error, optional repetitions, and finite-menu interpretation

For a single run, let the sine-clock state preparation, selected offset
ramp, and inverse QFT have certified errors `epsilon_clock`,
`epsilon_ramp`, and `epsilon_QFT`.  Use the maximum ramp error over the
finite classical branches (or an independently certified average if a
program explicitly proves it).  Unitary invariance and a hybrid bound give
a probability allowance at most twice their sum.  There is no `(P-1)`
factor on initial clock or ramp error.  The unchanged conditional logical
reserve is `1/100`.  Hence a sufficient whole-program bound is

\[
 p_{P,S}+2(\epsilon_{\rm clock}+\epsilon_{\rm ramp}
 +\epsilon_{\rm QFT})+1/100.
\]

For odd independent repetitions, draw a fresh uniform offset and prepare
the source and clock on every run.  The same head threshold can be reused.
The majority bound is the exact binomial upper tail

\[
 \sum_{j=(K+1)/2}^K {K\choose j}p_{P,S}^{j}(1-p_{P,S})^{K-j}.
\]

Add `2K*(epsilon_clock+epsilon_ramp+epsilon_QFT)` and the once-per-algorithm
logical reserve.  A head transcript is conditioned on throughout, so the
uniform bound also holds after averaging over transcripts.  Head queries
are charged once; quantum label queries are `K(P-1)`, without a factor
`S`.  Random classical bits, branch dispatch, shifted decoding, and
worst-branch native ramp resources must be explicit in the complete
program ledger.  This theorem by itself does not certify their native
implementation.

The finite menu supplies valid upper certificates, not an impossibility
result or a global optimum over all algorithms.  Both compared methods
use their own exact accepted bias.  No fitted values, input labels,
head-specific favorable phase ranges, or measured margins enter the menu.

## Exact target results

At `P=16384` and `Z=89698095963077/17592186044416`, the independently
produced statistical upper bounds are:

| S | Linear own-bias bound | Cubic/nodes own-bias bound |
|---:|---:|---:|
| 1 | 0.0987358327689939 | 0.0987358327689939 |
| 2 | 0.0493690624876207 | 0.0493690624876207 |
| 4 | 0.0299118377496155 | 0.0295896944378902 |
| 8 | 0.0220774004857858 | 0.0217062514868977 |
| 16 | 0.0185935563517810 | 0.0183268179044149 |

The identical first two displayed bounds result from the same conservative
integer ray-start envelopes despite different exact biases.  They are not
evidence that the underlying exact error functions are identical.  The
authoritative fractions and all 128 closed-cell witnesses are in
`results/random_sign/bound_discovery_n68_{linear,nodes}_P16384_S*.json`.
The complete fixed-menu report is `results/random_sign/bound_menu.json`.
Neither statistical figure alone includes the native allowance or logical
reserve.
