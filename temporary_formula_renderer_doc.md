$J_{ij} \in \reals $ is the bond strength between node $i$ and $j$, with $J_{ij} = J_{ji}$


$s_i(t) \in \{-1, 0, 1\}$ is the state of node $i$ at time $t$.

$h_i(t) \in \reals$ is the external field for node $i$ at time $t$.

$$h^{\text{eff}}_i(t) := \sum_{j\in \mathcal{N}(i)} J_{ij} s_j(t) + h_i(t) + \mu s_i(t\!-\!1)$$

$$h^{\text{eff}}_i(t) := \overbrace{\sum_{j\in \mathcal{N}(i)} J_{ij} s_j(t)}^{\text{local network}} + \underbrace{h_i(t)}_{\text{field}} + \overbrace{\mu s_i(t\!-\!1)}^{\text{consistency}}$$


$0 < \mu$: Memory coefficient

$0 < \beta$: Inverse temperature

Transition distribution:

Effective field:

Multiplicative interaction between $z$ and $h$ $\implies$ promote same sign (agreement)

$$P\big(s_i(t) \!=\! z \mid h^{\text{eff}}_i(t) \!=\! h\big) =  \frac{\exp(\beta zh)}{\sum_{z'\in\{-1, 0, 1\}} \exp(\beta z'h)}$$

$$m(t) := \frac{1}{N}\sum_{i=1}^N s_i(t) \in [-1, 1]$$