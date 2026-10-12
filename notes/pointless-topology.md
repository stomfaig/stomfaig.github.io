---
title: A glimpse into Pointless Topology
date: 2026-10-11
description: A quick tour of how to do point-set topology without points.
---

While reading a proof of soundness and completeness of intuitionistic propositional logic (IPC) for Kripke models, I encountered [pointless topology](https://en.wikipedia.org/wiki/Pointless_topology). The idea seemed exotic, so I wrote this post to work out how much further I wanted to pursue it.

### Spaces through their topology

Given a topological space $(X, O)$, we can consider the topology $O$ as an algebraic structure in its own right. Recall that $O$ being a topology means:

- $O \subseteq \mathcal{P}(X)$
- $\emptyset \in O, X \in O$
- $O$ is closed under finite intersections
- $O$ is closed under arbitrary unions.

These operations give $O$ a lattice structure. We translate this into lattice theory as follows:

::: {.definition}
  A _frame_ is a complete lattice that satisfies the _infinite distributive law_:
  $$a \wedge (\vee_{i \in I} b_i) = \vee_{i \in I} (a \wedge b_i)$$
  Define the category $\mathtt{Frm}$, whose objects are frames and whose morphisms preserve arbitrary joins and finite meets, including the empty join $0$ and the empty meet $1$.
:::

Meets are given by $\bigwedge_{i \in I} A_i = \operatorname{int}(\bigcap_{i \in I} A_i)$, which is just the usual intersection for $I$ finite. We can investigate the information encoded in the topology through the functor $\Omega:\mathtt{Top}^{op}\to\mathtt{Frm}$, which assigns to each topological space the frame of its open sets.

The functor $\Omega$ is _contravariant_: a continuous map $f: X \to Y$ induces a frame homomorphism $f^{-1}: \Omega(Y) \to \Omega(X)$. This leads us to define the category of _locales_ as $\mathtt{Loc}=\mathtt{Frm}^{op}$.

Can we recover the points of $X$ from its topology? Clearly, to have any chance of distinguishing them, we need a sufficient topological structure. This is exactly the [$T_0$ separation property](https://mathworld.wolfram.com/T0-SeparationAxiom.html): any two distinct points must differ in membership in some open set.

Our approach will be to try to recover point identity by the open sets that contain it. First, each $x\in X$ defines a frame homomorphism $\hat{x}: \Omega(X) \to \mathbb{S} = \{ 0 < 1 \}$, sending $U$ to $1$ exactly when $x\in U$. This suggests defining a _point_ of a frame $L$ as a frame homomorphism $L \to \mathbb{S}$. Write $\operatorname{Pt}(L)$ for the set of such points. Recovering $X$ requires every point of its frame to come from a unique point of $X$.

::: {.definition}
  A topological space $X$ is _sober_ if every frame homomorphism $\Omega(X) \to \mathbb{S}$ is induced by a unique point of $X$.
:::

The maps that we designated as points correspond exactly to _completely prime filters_ of $\Omega(X)$. Let's build this definition piece by piece:

- A filter $F\subseteq L$ is upward closed, contains $1$, and is closed under finite meets
- It is _proper_ if $0 \notin F$ (alternatively, $F \neq L$)
- It is _prime_ if $a \vee b \in F$ implies $a \in F$ or $b \in F$.
- A proper filter is _completely prime_ if $\bigvee_{i\in I}a_i\in F$ implies $a_i\in F$ for some $i\in I$.

Complete primeness is needed because frame homomorphisms preserve arbitrary joins. For a frame point $f$, the corresponding c-p filter is $f^{-1}(1)$; conversely, a completely prime filter defines a point by its characteristic function. We denote the filter induced by $x$ as $P_x$.

Now let's recover the topology. Given an open set $U\in\Omega(X)$, we have $x\in U$ if and only if $\hat{x}(U)=1$. For any frame $L$, the following sets form a topology on $\operatorname{Pt}(L)$, as we verify in the next section:
$$\{\{f\in\operatorname{Pt}(L):f(u)=1\}:u\in L\},$$
Denote this space by $\Sigma(L)$. The evaluation map is $\rho:X\to\Sigma(\Omega(X))$, $x\mapsto\hat{x}$. We then have the following result:

::: {.theorem}
  For a sober topological space $X$, the map $\rho: X \to \Sigma(\Omega(X))$ is a homeomorphism.
:::

The map $\rho$ is always continuous, and is injective when $X$ is $T_0$. Surjectivity is a further requirement: sobriety says precisely that $\rho$ is bijective, and the displayed description of the open sets then makes it a homeomorphism.

### Going the other way

The procedure in the previous section constructs a functor $\Sigma: \mathtt{Frm}^{op} \to \mathtt{Top}$. Note that the sets $U_a=\{f\in\operatorname{Pt}(L):f(a)=1\}$ defined above satisfy the following equations:
$$U_{\bigvee_i a_i}= \bigcup_i U_{a_i}, \quad U_{a \wedge b} = U_a \cap U_b $$
Together with $U_0=\emptyset$ and $U_1=\operatorname{Pt}(L)$, these equations show that the sets form a topology.[^1]

The frame homomorphism $\theta: L \to \Omega(\Sigma(L))$, given by $a \mapsto U_a$, is surjective. When it is also injective, $L$ is called _spatial_. Spatiality can also be expressed as follows: whenever $a \nleq b$, there is a point $f$ with $f(a)=1$ and $f(b)=0$.

We next characterize spatiality using prime elements, then give a sufficient condition using compact elements.

### A complementary point of view

So far we have studied points through the open sets they belong to. We can instead consider the opens they _do not_ belong to, using ideals:

::: {.definition}
  An _ideal_ $I$ of a bounded distributive lattice contains $0$, is downward closed, and is closed under finite joins. It is _prime_ if it is proper and $a\wedge b\in I$ implies $a\in I$ or $b\in I$.
:::

For a frame $L$, we consider prime ideals that are additionally closed under arbitrary joins. Taking complements gives a bijection between these ideals and completely prime filters. In particular, the opens that do not contain a given point $x$ form such an ideal.

We can relate such ideals with special elements in them: arbitrary-join closure gives $p_I = \bigvee I \in I$. Downward closure then gives
$$a \in I \iff a \leq p_I.$$

Further, $p_I$ is prime in the following sense:

::: {.definition}
  An element $p$ of a frame $L$ is _prime_ if $p < 1$ and
  $$a \wedge b \leq p \quad \Longrightarrow \quad a \leq p \text{ or } b \leq p.$$
:::

Indeed, $a \wedge b \leq p_I$ means $a \wedge b \in I$. Primeness of $I$ gives $a \in I$ or $b \in I$, hence $a \leq p_I$ or $b \leq p_I$. Properness gives $p_I < 1$.

Conversely, a prime element $p$ defines a frame point $f_p$ by $f_p(a)=0$ when $a \leq p$, and $f_p(a)=1$ otherwise. Primeness ensures preservation of finite meets, while $\bigvee_i a_i \leq p$ holds exactly when every $a_i \leq p$, ensuring preservation of arbitrary joins.

The canonical example of prime ideals closed under arbitrary joins is given by the open sets that do not contain a fixed point $x \in X$. Since prime ideals closed under arbitrary joins correspond to frame points, we get the following alternative characterisation of sobriety:

::: {.theorem}
  A topological space $X$ is sober if and only if every prime element of $\Omega(X)$ has the form $X \setminus \overline{\{x\}}$ for a unique $x \in X$.
:::

### Spatiality via prime elements

Write $P(L)$ for the set of prime elements of $L$. The correspondence with frame points gives the following criterion:

::: {.theorem}
  A frame $L$ is spatial if and only if, for every $a\in L$,
  $$a = \bigwedge \{ p \in P(L) : a \leq p \}$$
:::

Suppose $a \nleq b$ and $b$ is the meet of the primes above it. Some such prime $p$ satisfies $a \nleq p$. Its point $f_p$ has $f_p(a)=1$ and $f_p(b)=0$, so points separate elements and $L$ is spatial.

Conversely, suppose $L$ is spatial and let $q$ be the meet of the primes above $b$. Certainly $b \leq q$. If $q \nleq b$, a separating point gives a prime $p$ with $b \leq p$ but $q \nleq p$, contradicting the definition of $q$. Thus $q=b$.

### Compact elements and spatiality

We also give another sufficient condition of spatiality using _compact_ elements.

In topology, one usually defines a compact set as one for which "every open cover has a finite subcover." For an open set $U$, this can be expressed entirely in the frame of opens: whenever $U \leq \bigvee_{i \in I} U_i$, already $U \leq \bigvee_{i \in J} U_i$ for some finite $J \subseteq I$. This suggests the following definition.

::: {.definition}
  An element $c$ of a frame $L$ is _compact_ if, for every family $(a_i)_{i \in I}$,
  $$c \leq \bigvee_{i \in I} a_i \quad \Longrightarrow \quad c \leq \bigvee_{i \in J}a_i \text{ for some finite } J \subseteq I.$$
  A frame is _algebraic_ if every element is the join of the compact elements below it:
  $$a= \bigvee \{c \leq a : c \text{ is compact}\}.$$
:::

For example, $\mathcal{P}(X)$ is algebraic: its compact elements are exactly the finite subsets, and every subset is their union.

Algebraicity suffices for spatiality.

::: {.theorem}
  Assuming Zorn's lemma, every algebraic frame is spatial.
:::

Suppose $a \nleq b$. Since $a$ is a join of compact elements, there is a compact $c \leq a$ with $c \nleq b$. Consider
$$M=\{x \in L : b \leq x \text{ and } c \nleq x\}.$$
This set contains $b$. Every nonempty chain in $M$ has its join in $M$: otherwise compactness would put $c$ below a finite join of chain elements, hence below one of them. The empty chain has upper bound $b$. Zorn's lemma therefore gives a maximal $p\in M$.

We claim that $p$ is prime. If $u \nleq p$ and $v \nleq p$, maximality implies $c \leq p \vee u$ and $c \leq p \vee v$. Distributivity gives
$$c \leq (p \vee u) \wedge(p \vee v) = p \vee (u \wedge v).$$
If $u \wedge v \leq p$, this would contradict $c \nleq p$. Thus $u \wedge v \nleq p$, proving primeness. Also $p < 1$, since $c \nleq p$.

The corresponding frame point $f_p$ satisfies $f_p(b)=0$ and $f_p(a)=1$, because $b \leq p$ but $c \leq a$ and $c \nleq p$. Points therefore separate every pair $a \nleq b$, so $\theta$ is injective and $L$ is spatial.


### Final remarks

There are many aspects of the topic that were not discussed here. One further direction is the correspondence between coherent frames and spectral spaces. See [Kock's notes](https://mat.uab.cat/~kock/cat/spec.pdf) for an introduction.
The viewpoint also has advantages beyond reformulating familiar topology. For example, the localic version of Tychonoff's theorem proves that products of compact locales are compact without the axiom of choice. Recovering a space of points is a separate matter: the resulting locale need not be spatial. [Picado and Pultr's notes](https://www.mat.uc.pt/~picado/publicat/LLNchapter.pdf), Section 7.5, discuss this distinction.

For this first glimpse, the question was how much of a space survives when we keep only its open sets. Sobriety tells us when we recover its points; spatiality tells us when points recover the frame, which gives a basic feel for what is going on here.

### References

- Peter T. Johnstone, [*The point of pointless topology*](https://doi.org/10.1090/S0273-0979-1983-15080-2), *Bulletin of the American Mathematical Society* 8 (1983), 41–53.
- Peter T. Johnstone, [*Stone Spaces*](https://books.google.com/books?id=CiWwoLNbpykC), Cambridge Studies in Advanced Mathematics 3, Cambridge University Press, 1982.
- Joachim Kock, [*Remarks on spectra, supports, and Hochster duality*](https://mat.uab.cat/~kock/cat/spec.pdf), 2007.
- Jorge Picado and Aleš Pultr, [*Notes on point-free topology*](https://www.mat.uc.pt/~picado/publicat/LLNchapter.pdf).

[^1]: Previously, we checked how this construction recovers the opens of an existing space. Here we verify that it defines a topology for an arbitrary frame, without a space to start from.


---

*If you spot an error or have a question, feel free to open an issue or send me an email.*
