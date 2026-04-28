# Introducing the notation and formalising the problem

### Probability Space

We work throughout on a filtered probability space \((\Omega, \mathcal{F}, \mathbb{F}, \mathbb{P})\), where \(\Omega\) is the sample space of all possible market scenarios, \(\mathcal{F}\) is the \(\sigma\)-algebra of observable events, \(\mathbb{F} = (\mathcal{F}_t)_{t \geq 0}\) is a filtration representing the information available up to time \(t\), and \(\mathbb{P}\) is the physical (real-world) probability measure.


### Assets and Prices

Let \(\mathcal{I} = \{1, \ldots, N\}\) denote the universe of \(N\) assets (stocks) and \(\mathcal{T} = \{0, 1, \ldots, T\}\) denote discrete trading dates (here, business days).

**Definition  — Price Process** 
For each asset \(i \in \mathcal{I}\), the price process is a sequence of strictly positive random variables

\[P_i = (P_{i,t})_{t \in \mathcal{T}}, \quad P_{i,t} : \Omega \to \mathbb{R}_{>0},\]
where each \(P_{i,t}\) is \(\mathcal{F}_t\)-measurable. We always work with total-return adjusted prices, meaning \(P_{i,t}\) accounts for dividends and stock splits as if they were reinvested.

**Definition — Simple Return**

\[R_{i,t} := \frac{P_{i,t} - P_{i,t-1}}{P_{i,t-1}} = \frac{P_{i,t}}{P_{i,t-1}} - 1\]
The simple return is the fractional change in wealth from holding asset \(i\) from \(t-1\) to \(t\).

**Definition — Log Return**
\[r_{i,t} := \log \frac{P_{i,t}}{P_{i,t-1}} = \log(1 + R_{i,t})\]
Log returns are additive across time: the log return from \(s\) to \(t\) is \(\sum_{\tau=s+1}^{t} r_{i,\tau}\). We use log returns throughout.


### Portfolios

**Definition — Portfolio Weight Vector**s
A portfolio at time \(t\) is a vector \(\mathbf{w}_t = (w_{1,t}, \ldots, w_{N,t})^\top \in \mathbb{R}^N\), where each \(w_{i,t}\) is \(\mathcal{F}_t\)-measurable and represents the fraction of total capital allocated to asset \(i\). We impose:

\[\sum_{i=1}^{N} w_{i,t} = 1 \quad \text{(fully invested)}\]
Short positions are allowed: \(w_{i,t} < 0\) means we have borrowed and sold asset \(i\).

**Definition — Dollar-Neutral Portfolio**
A portfolio is dollar-neutral (or market-neutral) if

\[\sum_{i=1}^{N} w_{i,t} = 0.\]
The long leg satisfies \(\sum_{w_{i,t}>0} w_{i,t} = L\) and the short leg \(\sum_{w_{i,t}<0} w_{i,t} = -L\) for some \(L > 0\). The portfolio requires no net capital: losses on one leg are financed by gains (or short proceeds) on the other.

*Why dollar-neutral?*
A dollar-neutral portfolio is (approximately) insensitive to broad market movements. If the whole market falls 2%, a well-hedged dollar-neutral portfolio is largely unaffected. This isolates stock-selection skill from market timing — a far harder and less reliable bet. See the notebook: ______ to for further motivation.

**Definition — Portfolio Return**
The one-period return of portfolio \(\mathbf{w}_t\) realised over \([t, t+1]\) is

\[R_t^{\text{port}} = \sum_{i=1}^{N} w_{i,t} \cdot R_{i,t+1} = \mathbf{w}_t^\top \mathbf{R}_{t+1},\]
where \(\mathbf{R}_{t+1} = (R_{1,t+1}, \ldots, R_{N,t+1})^\top\). Note the critical timing: weights \(\mathbf{w}_t\) are set using only \(\mathcal{F}_t\)-measurable information, but returns \(\mathbf{R}_{t+1}\) are realised at \(t+1\).

# The cross-sectional prediction problem

Rather than forecasting the absolute level of \(\mathbf{R}_{t+1}\), we focus on relative performance. Define the cross-sectionally demeaned forward return:

**Definition 5.1 — Demeaned Forward Return (Target Variable)**
\[y_{i,t} := \left(\sum_{\tau=1}^{H} r_{i, t+\tau}\right) - \frac{1}{N}\sum_{j=1}^{N}\sum_{\tau=1}^{H} r_{j, t+\tau},\]
where \(H = 21\) trading days \(\approx 1\) month. The subtracted term is the equal-weighted market return over the horizon. Thus \(y_{i,t} > 0\) means asset \(i\) outperformed the average, and \(\sum_i y_{i,t} = 0\) by construction.


**Definition 5.2 — Feature Vector**
For each asset \(i\) and date \(t\), we construct a feature vector

\[\mathbf{x}_{i,t} = \varphi\!\left((r_{i,s}, \text{Vol}_{i,s}, \ldots)_{s \leq t-1}\right) \in \mathbb{R}^p,\]
where \(\varphi\) is a deterministic measurable function of past data only. Crucially, \(\mathbf{x}_{i,t}\) is \(\mathcal{F}_{t-1}\)-measurable — it uses no information from time \(t\) or later. This is the no-lookahead constraint.


**The prediction problem is then**: find a function \(f : \mathbb{R}^p \to \mathbb{R}\), estimated from data \(\{(\mathbf{x}_{i,s}, y_{i,s})\}_{s < t}\), such that

\[\hat{y}_{i,t} = f(\mathbf{x}_{i,t}; \hat{\theta})\]
is a useful predictor of \(y_{i,t}\). We then construct a dollar-neutral portfolio by going long stocks with high \(\hat{y}_{i,t}\) and short stocks with low \(\hat{y}_{i,t}\).

