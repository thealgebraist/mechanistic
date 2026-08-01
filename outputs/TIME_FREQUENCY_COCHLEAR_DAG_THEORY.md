# Time-varying frequency and cochlear probabilistic DAGs

## Result

There are closed-form mathematical descriptions of time-parametrized frequency, but they describe different objects. The cleanest basis for a positive probabilistic DAG is a filterbank energy measure. The cleanest basis for explicit frequency trajectories is an amplitude–phase model or synchrosqueezed transform. The most biologically meaningful basis is a time-unrolled cochlear state-space cascade such as CAR-FAC.

None of these representations is automatically a semantic decompilation of Whisper. A graph becomes an equivalent frontend only after proving that its transport into the model's 80 Mel coordinates preserves the downstream conditional token law.

## 1. Where filter theory comes from

Classical filter theory sits at the intersection of harmonic analysis, linear systems, complex analysis, and control theory. A linear time-invariant filter is any convolution operator

\[
y(t)=(h*x)(t)=\int_{-\infty}^{\infty}h(\tau)x(t-\tau)\,d\tau.
\]

Fourier transformation diagonalizes this operator: \(Y(\omega)=H(\omega)X(\omega)\). For causal rational filters, the Laplace-domain transfer function \(H(s)\) is represented by poles and zeros and can be realized by a finite-dimensional state equation

\[
\dot z(t)=Az(t)+Bx(t),\qquad y(t)=Cz(t)+Dx(t).
\]

This is why a filterbank already has a graph/program interpretation: each section is a small state machine, and a cascade is composition of those machines.

## 2. Closed-form models of frequency changing with time

### 2.1 Amplitude–phase and polynomial-phase atoms

For an analytic component

\[
x_k(t)=A_k(t)e^{i\phi_k(t)},
\]

the angular instantaneous frequency is \(\omega_k(t)=\phi'_k(t)\), or \(f_k(t)=\phi'_k(t)/(2\pi)\) in hertz. This is exact once a component and its phase have been chosen. Important finite descriptions include:

\[
\phi(t)=\sum_{r=0}^{d}a_rt^r
\]

for polynomial-phase signals, and the linear chirplet

\[
x(t)=A e^{-(t-t_0)^2/(2\sigma^2)}
e^{i(\omega_0(t-t_0)+\beta(t-t_0)^2/2)},
\qquad \omega(t)=\omega_0+\beta(t-t_0).
\]

A DAG can quotient the continuous parameter space \((t_0,\sigma,\omega_0,\beta,A)\) into cells. Its nodes then mean sets of chirp trajectories rather than fixed frequency bands.

The limitation is identifiability: a general waveform has many possible component decompositions. Instantaneous frequency is therefore not a unique property of an arbitrary real signal without assumptions on component separation and regularity.

### 2.2 Gabor/STFT energy

The short-time Fourier transform is

\[
V_gx(t,\omega)=\int x(\tau)\overline{g(\tau-t)}e^{-i\omega\tau}\,d\tau.
\]

Its spectrogram \(|V_gx(t,\omega)|^2\) is nonnegative, so a finite partition \(B_j\) of frequency produces an immediate probability mass

\[
E_{t,j}=\int_{B_j}|V_gx(t,\omega)|^2d\omega,
\qquad p_{t,j}=\frac{E_{t,j}}{\sum_\ell E_{t,\ell}}.
\]

This is the mathematical pattern used by the Mel, uniform-subband, and sparse-resonator experiments. Resolution is limited by the window's time–frequency uncertainty.

### 2.3 Wigner–Ville distributions

The quadratic distribution

\[
W_x(t,\omega)=\int x(t+\tau/2)\overline{x(t-\tau/2)}e^{-i\omega\tau}\,d\tau
\]

has sharp time–frequency localization and correct marginals, but it can be negative and generates cross-terms between components. It is a quasiprobability, not directly a categorical probability. Positive smoothing yields a spectrogram-like distribution but sacrifices resolution.

### 2.4 Wavelets and synchrosqueezing

The continuous wavelet transform

\[
W_x(a,b)=a^{-1/2}\int x(t)\overline{\psi((t-b)/a)}\,dt
\]

uses scale \(a\), with characteristic frequency approximately proportional to \(1/a\). For separated, slowly varying AM–FM components, synchrosqueezing estimates local frequency from a phase derivative and reassigns wavelet mass toward frequency ridges. Daubechies, Lu, and Wu define an intrinsic-mode-type function class and prove recovery results under explicit separation and slow-variation assumptions. This gives a principled route to a small “trajectory DAG,” but not a guarantee for arbitrary audio.

## 3. Cochlear mathematics

### 3.1 Place–frequency map

The cochlea is a nonuniform traveling-wave medium. Greenwood's empirical map has the form

\[
f(x)=A(10^{ax}-k),
\]

where \(x\) is normalized or physical distance and the constants depend on species and coordinate convention. This turns a cochlear place interval into an interpretable frequency-set node.

Glasberg and Moore's normal-hearing equivalent rectangular bandwidth approximation is commonly written

\[
\operatorname{ERB}(f)=24.7(1+4.37f/1000)\ \mathrm{Hz}.
\]

ERB spacing gives more channels per hertz at low frequencies, matching auditory resolution better than equal-Hz spacing.

### 3.2 Gammatone approximation

The classic auditory-filter impulse response is

\[
g_j(t)=a_jt^{n-1}e^{-2\pi b_jt}\cos(2\pi f_jt+\varphi_j)\mathbf 1_{t\ge0},
\]

usually with order \(n=4\) and bandwidth tied to ERB. Rectification, compression, and low-pass smoothing can be appended as an inner-hair-cell approximation. This is explicit and inexpensive, but it is a parallel filterbank approximation, not a mechanical traveling-wave model.

### 3.3 CAR-FAC

Lyon's cascade of asymmetric resonators with fast-acting compression is closer to cochlear wave mechanics. In its linearized continuous form, one asymmetric stage has a pole–zero transfer function

\[
H_j(s)=
\frac{s^2/\omega_{z,j}^2+2\zeta_{z,j}s/\omega_{z,j}+1}
     {s^2/\omega_{p,j}^2+2\zeta_{p,j}s/\omega_{p,j}+1}.
\]

The zeros sit above the poles in frequency. Sections are cascaded from high to low characteristic frequency, approximating basal-to-apical wave propagation. Outer-hair-cell feedback changes pole damping according to local activity; half-wave/detection and capacitor dynamics model inner-hair-cell transduction; spatially and temporally smoothed activity drives a multi-timescale AGC loop.

A compact typed state for section \(j\) at sample \(t\) is

\[
S_{t,j}=(z^1_{t,j},z^2_{t,j},z^a_{t,j},r_{t,j},g_{t,j},v^{IHC}_{t,j},a^{AGC}_{t,j,1:m}).
\]

The update has the dependency pattern

\[
(S_{t,j},y_{t,j-1},A_{t,\mathcal N(j)})
\longmapsto (S_{t+1,j},y_{t,j},A_{t+1,j}),
\]

where \(y_{t,j-1}\) is the preceding cascade section and \(\mathcal N(j)\) are neighboring cochlear places used by AGC smoothing. The physical model is recurrent, but its finite execution is a DAG after unrolling by time: all edges increase either the section index within one sample or the time index.

## 4. The probabilistic quotient DAG

For any frontend, let \(C_i\) be its fine channels and let \(q:C_i\to B_j\) merge them into explicit quotient nodes. Define a positive activity measure

\[
\mu_t(B_j)=\sum_{i:q(C_i)=B_j}E_{t,i},\qquad
p_t(j)=\frac{\mu_t(B_j)}{\mu_t(\Omega)}.
\]

Then quantize \(p_t\) or log-energy into finite algebraic data:

```
Frontend = Mel | UniformBand | Goertzel | WaveletPacket | CochlearCARFAC
Node     = FrequencySet Finset[DFTBin]
         | PacketSet Finset[WaveletPacket]
         | CochlearPlaceSet Finset[CARSection]
State L  = Vector 80 (Fin L)
```

For a finite recording, nodes \((t,j)\) and time-forward transitions form a finite probabilistic DAG. For arbitrary-duration audio, the correct object is a finite parametric graph schema or probabilistic transducer, not one finite DAG.

The strongest exact quotient merges states \(u,v\) only when their downstream conditional transcript laws agree:

\[
u\sim v\quad\Longleftrightarrow\quad
\forall s\in\mathrm{Token}^*,\quad
P(s\mid u)=P(s\mid v).
\]

An approximate quotient can replace equality by total variation, Wasserstein distance, or a task-specific distortion. Data processing then says that a fixed Markov kernel cannot increase an established divergence. The missing proof obligation is crucial: one must first certify a uniform divergence bound between the alternative frontend and the Mel representation at the Whisper interface. Similar-looking frequency partitions do not supply that bound.

## 5. What the experiment establishes

The test executes five explicit 80-node frontends on the same waveform. The CAR-FAC lane uses Google's official NumPy implementation at pinned commit `c74663cc7d05713ae2f2308765eb040530a81c7f`, with 81 physical sections, two-capacitor IHC dynamics, and closed-loop AGC. Adjacent places are quotiented to 80 nodes and the mean-squared neural activity in each frame is normalized into a probability mass.

This establishes a reproducible cochlear probabilistic DAG construction and measures its behavior through Whisper. It does **not** establish that the cochlear representation and Mel representation are universally equivalent. In fact, the poor uncalibrated transcript is a concrete counterexample to naïve drop-in equivalence on this recording.

## 6. Recommended next theorem and experiment

Learn a small nonnegative transport matrix \(T\in\mathbb R_+^{80\times80}\) from cochlear place mass to Mel mass, constrained by column normalization and locality along frequency. Then certify on a bounded waveform domain:

\[
\sup_x D_{JS}(T p^{CARFAC}(x),p^{Mel}(x))\le\varepsilon_F.
\]

If the fixed Whisper suffix is shown to be \(L\)-Lipschitz in a compatible metric, the transcript-law error is bounded by a transported term such as \(L\varepsilon_F\). Convex optimization can find \(T\); interval or convex relaxations must certify the uniform bound. Until that second step is complete, the result is an empirical adapter, not a proof.

## Primary sources

- D. Gabor, “Theory of communication,” *Journal of the IEE* 93(26), 1946, DOI [10.1049/ji-3-2.1946.0074](https://doi.org/10.1049/ji-3-2.1946.0074).
- I. Daubechies, J. Lu, and H.-T. Wu, “Synchrosqueezed Wavelet Transforms,” [arXiv:0912.2437](https://arxiv.org/abs/0912.2437).
- R. D. Patterson et al., “An efficient auditory filterbank based on the gammatone function,” 1987; [Cambridge report index](https://www.pdn.cam.ac.uk/other-pages/cnbh/reissued-conference-papers).
- B. R. Glasberg and B. C. J. Moore, “Derivation of auditory filter shapes from notched-noise data,” *Hearing Research* 47, 1990, DOI [10.1016/0378-5955(90)90170-T](https://doi.org/10.1016/0378-5955(90)90170-T).
- D. D. Greenwood, “A cochlear frequency-position function for several species—29 years later,” *JASA* 87, 1990, DOI [10.1121/1.399052](https://doi.org/10.1121/1.399052).
- R. F. Lyon, “Cascades of two-pole–two-zero asymmetric resonators are good models of peripheral auditory function,” *JASA* 130, 2011, [author PDF](https://www.dicklyon.com/tech/Hearing/Lyon2011_JASMAN13063893_1.pdf).
- Google CAR-FAC authors, [official implementation](https://github.com/google/carfac), pinned here to commit `c74663cc7d05713ae2f2308765eb040530a81c7f`.
