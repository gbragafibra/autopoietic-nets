# Autopoietic Networks

A type of cellular automaton based on the concept of (autopoiesis)[https://en.wikipedia.org/wiki/Autopoiesis] is explored, as introduced by (Maturana)[https://en.wikipedia.org/wiki/Humberto_Maturana] and (Varela)[https://en.wikipedia.org/wiki/Francisco_Varela] in [Autopoiesis and Cognition: The Realization of the Living](https://doi.org/10.1007/978-94-009-8947-4), and also a bit developed on in Principles of biological autonomy(1979) by Varela.

Autopoiesis can be viewed as the capacity of an organism to maintain (organizational closure)[https://doi.org/10.1016/j.jtbi.2015.02.029] or closure of constraints. These very briefly mean that any process (and respective constraints on it) needs to generate atleast another constraint. Here, processes and relations between them evolve in order to maintain such closure. Through this lens, any new adaptation or behaviour taken by an organism can be seen as a compensation to a pertubation (i.e. self-replication).

### Details of the network

A network with `N x N` units is initialized, with each unit's state $∈ \{0, 1\}$ being randomly assigned. Over each iteration (with a max of `N_iter`), there's a logic gate randomly chosen from a set (examples with `AND`, `OR` and `XOR`), for the update of each unit's state according to its neighbors' states (`extended` option to also include neighbors in the diagonals).

If a unit maintains its state after an iteration, regardless of its neighbors, its $\Phi$ value (which is supposed to mimick closure) increases by 1. Moreover, if the states respective to consecutive iterations are different, $\Phi$ is reset to 0. If $\Phi \geq \epsilon$, then the neighbors take the value of the main unit's state, and an ensemble is formed. With regard to this, $\epsilon$ can be fixed (`fix = True`) or evolve over time (`fix = False`) with 
$$
ε = \textrm{int}\left(\frac{\textrm{N_iter} * (1 - \textrm{H}(S))\right}{k})
$$
where $\textrm{N_iter}$ is the number of iterations chosen, $\textrm{H}(S)$ the entropy of the net, and $k$ being an assigned integer. Additionally, there are two other options: `geq_cond = True` for $\Phi \geq \epsilon$ and `geq_cond = False` for $\Phi = \epsilon$.


### Some results

<iframe width="560" height="315" src="https://www.youtube.com/watch?v=Az061cv_s7A" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

