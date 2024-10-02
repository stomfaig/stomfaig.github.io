---
layout: post
title: "Covering spaces"
mathjax: true
categories: misc
published: false
---


*This post is an adoptation of Chapter 1.3 of Allan Hatcher's Algebraic Topology. As such, it contains no original work apart from the style of the exposition, and likely errors.*

---

Let $X$ be a topological space. $p: \tilde{X} \to X$ is a **covering space** for $X$, if there is an open cover $\{U_\alpha\}$ for $X$ so that for each $\alpha$, $p^{-1}(U_\alpha)$ is the disjoint union of open sets in $\tilde{X}$, and each of them is mapped homeomorphically onto $U_\alpha$.  
Given a map $f: Y \to X$, a **lift** of $f$ is a map $\tilde{f}: Y \to \tilde{X}$, so that $p \tilde{f} = f$. We have the following proposition.

**Proposition 1.** Given a covering space $p: \tilde{X} \to X$, a homotopy $f_t : Y \to X$, and a map $\tilde{f_0}$ lifting $f_0$, then there exists a unique homotopy $\tilde{f_t}: Y \to \tilde{X}$ of $\tilde{f_0}$ which lifts $f_t$.

**Proof.** *Coming soon.* $\blacksquare$

**Remark.** This is really convenient, since lifting $f_0$ is in general is not too complicated. Note however, that the lift of $\tilde{f_t}$ we obtain is not at all guaranteed, and in practice is usually not behaving very well -- at least at first glance.

**Remark.** As consequence of the uniqueness property of lifts, if find two lifts of the same homotopy (one of which behaves nicely, and the other we are trying to prove the properties of), we can immediately conclude that the two are the same. This in general can be a really useful tool, see for example Proposition 3.

We can also consider some special cases of this.

**Path lifting property.** Take $Y$ to be a single point in the above. Given a path $f: I \to X$, and a a lift $\tilde{x_0}$ of the starting point, then there is a unique path $\tilde{f}: I \to \tilde{X}$ lifting $f$ and starting at $\tilde{x_0}$.  
Now, suppose that we are lifting the constant path $f(t) = x_0, \ \forall t \in I$. We can easily find a lift: $\tilde{f}(t)=\tilde{x_0}$. By the uniqueness property, however this is the unique lift of the constant path, and thus we conclude:

**Corollary.** The lift of a constant path is constant.

Recall that given a map $f: (Y, y_0) \to (X, x_0)$, by $f_*$ we denote the homomorphism that $f$ induces $\pi_1(Y, y_0) \to \pi_1(X, x_0)$.

**Proposition 2.** The map $p_* :\pi_1(\tilde{X}, \tilde{x_0}) \to \pi_1(X, x_0)$ induced by a covering space is injective. The image subgroup $p_*(\pi_1(\tilde{X}, \tilde{x_0}))$ in $\pi_1(X, x_0)$ consists of the homotopy classes of loops in $X$ based at $x_0$ whose lifts to $\tilde{X}$ starting at $\tilde{x_0}$ are loops.

**Remark.** First this can seem a bit unintuitive. After all, if $\tilde{X}$ is like $X$ but with more sheets, should it not be the case that one loop can have multiple images? Note however that $\tilde{x_0}$ is fixed. Then, since lifts are unique, injectivity follows immediately.

**Proof.** Conisder $\tilde{f}: I \to \tilde{X}$, a loop in $\tilde{X}$ starting at $\tilde{x_0}$, so that $[\tilde{f}] \in \ker p_*$. Then, there is a homotopy $f_t: I \to X$ so taking $f = p \tilde{f}$ to the constant trivial loop $f_1$. By Proposition 1, $f_t$ can be lifted to the starting point $\tilde{f}$ in a unique way. Note also, that since $f_1$ is constant, $\tilde{f_1}$ is a constant loop at $\tilde{x_0}$, implying that $[\tilde{f}] = 0$ in $\pi_1(\tilde{X}, \tilde{x_0})$. This proves injectivity.  
One implication in the second part is clear. Now suppose that $[f] \in p_*(\pi_1(X, x_0))$. Then $f$ is homotopic to a curve which is the image $p_*$. Any such curve has a unique lift to its pre-image, which thus concludes the proof. $\blacksquare$

It is often useful to lift maps that are not homotopies:
 
**Definition.**(Locally connected space) We say that a space $Y$ is locally path connected, if for each $y \in Y$ and open neighborhood $U$ of $y$ there is an open neighborhood $V \subseteq U$ of $y$ so that $V$ is path connected.

**Proposition 3.** Suppose that given a covering space $p: (\tilde{X}, \tilde{x_0}) \to (X, x_0)$, and a map $f: (Y, y_0) \to (X, x_0)$ with $Y$ path connected and locally connected. Then a lift $\tilde{f}: (Y, y_0) \to (\tilde{X}, \tilde{x_0})$ exists *iff* $f_*(\pi_1(Y, y_0)) \subset p_*(\pi_1(\tilde{X}, \tilde{x_0}))$.

**Remark.** One can think of this condition, as lifts exists if the space $Y$ is topologically not more complicated than the covering space $\tilde{X}$.

**Proof.** The forward direction is trivial.
For the converse, let $y \in Y$, and $\gamma$ a path from $y_0$ to $y$ (which exists since $Y$ is path connected). The path $f\gamma$ in $X$ starts from $x_0$ and thus has a unique lift starting at $\tilde{x_0}$. Define $\tilde{f}(y) = \tilde{f \gamma}(1)$. To show that this is well-defined, consider another curve $\gamma'$ from $y_0$ to $y$. Then $h_0 = (f \gamma') \circ(f \gamma)$ is a loop in $X$ starting at $x_0$. Since $f_*(\pi_1(Y, y_0)) \subset p_*(\pi_1(\tilde{X}, \tilde{x_0}))$, there is a homotopy $f_t:I \to X$ taking $h_0$ to $h_1$, where $h_1$ can be lifted to a loop $\tilde{h_1}$ at $\tilde{x_0}$ (see Proposition 2). We also see that the homotopy taking $h_0$ to $h_1$ can be lifted too, and since $\tilde{h_1}$ is a loop at $\tilde{x_0}$, $\tilde{h_0}$ must also be a loop at $\tilde{x}_0$.  
By the uniqueness of the lifted paths, the first half of $\tilde{h_0}$ is $\tilde{f\gamma'}$, and the second half is $\tilde{f \gamma}$ backwards. From this we see that $\tilde{f \gamma'}(1) = \tilde{f \gamma}(1)$, this the mapping $\tilde{f}$ is in fact well-defined.  
Now we claim that $\tilde{f}$ is continuous. Let $U \subset X$ be a neighborhood of $f(y)$, having a lift $\tilde{U} \subset \tilde{X}$ containing $\tilde{f}(y)$ (i.e. a sheet that contains it), so that $p: \tilde{U} \to U$ is a homeomorphism. Choose $V$ to be a path-connected neighborhood of $y$, with $f(V) \subset U$.  
For paths from $y_0$ to $y' \in V$, we can take a fixed path $\gamma$ from $y_0$ to $y$, followed by paths $\eta$ in $V$ from $y$ to points $y'$. Then, $(f \gamma) \circ (f \eta)$ has lift $\tilde{(f \gamma)} \circ \tilde{(f \eta)}$, where $\tilde{(f \eta)} = p^{-1}f \eta$, and hence $\tilde{f}(V) \subset \tilde{U}$. $\blacksquare$

Previously we've seen that if the starting point of a homotopy can be lifted, then the whole homotopy can be lifted in an unique way. We have the following, stronger version of the unique homotopy lifting property:

**Proposition 4.** Given a covering space $p: \tilde{X} \to X$, and a map $f: Y\to X$ with two lifts $\tilde{f_1}, \tilde{f_1}:Y \to \tilde{X}$ that agree at one point of $Y$, then if $Y$ is connected, they must agree on all of $Y$.

**Proof.** For any $y \in Y$, let $U$ be an open neighborhood of $f(y)$ in $X$, for which $p^{-1}(U)$ is a disjoint union of open sets $\{ \tilde{U_\alpha} \}_\alpha$, which are all homeomorphic to $U$ via $p$. Let $\tilde{U_1}$ and $\tilde{U_2}$ be the $\tilde{U_\alpha}$'s containing $\tilde{f_1}(y)$ and $\tilde{f_2}(y)$ respectively.  
Since $\tilde{f_i}$ are continuous, there is a neighborhood $N$ of $y$ so that $\tilde{f_1}(N) \subset \tilde{U}_1$, $\tilde{f_2}(N) \subset \tilde{U}_2$, i.e. $\tilde{f_i}$ is locally in the same $\tilde{U_\alpha}$.  
If $\tilde{f_1}(y) \neq \tilde{f_2}(y)$, then $\tilde{U_1} \neq \tilde{U_2}$, and thus they are disjoint, and $\tilde{f_1} \neq \tilde{f_2}$ on $N$, i.e. the $\tilde{f_i}$ are locally not equal. Since $Y$ is connected this implies that the tw functions are not equal anywhere on $Y$.  
If $\tilde{f_1}(y) = \tilde{f_2}(y)$, then similarly the two functions are locally equal, and since $Y$ is connected, we have $\tilde{f_1} = \tilde{f_2}$ on $Y$. $\blacksquare$

**Definition.** We says that a space $X$ **semilocally simply-connected**, if each $x \in X$ has a neighborhood $U$ such that the homomorphism $\pi_1(U, x) \to \pi_1(X, x)$ induced by the inclusion $U \hookrightarrow X$ is trivial.

**Proposition 5.** Suppose that $X$ is a path-connected, locally path-connected and semilocally simply connected. Then $X$ has a simply-connected covering space.

**Proof.** Let $x_0 \in X$ be a base-point, and consider
$$\tilde{X} = \{ [y] : \text{$\gamma$ is a path in $X$ starting at $x_0$} \}$$
The map $p: \tilde{X} \to X$ given by $[\gamma] \mapsto \gamma(1)$ is well-defined. Since $X$ is path connected, $p$ is clearly surjective.  
Let $\mathcal{U}$ be the collection of path-connected open sets $U \subset X$ so the homomorphism of fundamental groups is induced by the inclusion map is trivial. We claim that $\mathcal{U}$ is a basis for the topology of $X$, which we prove in a lemma after the main proof. Given $U \in \mathcal{U}$, set
$$U_{[\gamma]} = \{ [\gamma \cdot \eta] : \text{$\eta$ is a path in $U$ with $\nu(0) = \gamma(1)$}\} $$
Since each $U \in \mathcal{U}$ is path connected, $p: U_{[\gamma]} \to U$ is surjective. Furthermore, $p$ is also injective, since any two paths connecting $\gamma(1)$ to $y \in U$ are homotopic.  
Note also that, $U_{[\gamma]} = U_{[\gamma']}$ if $[\gamma'] \in U_{[\gamma]}$, which implies that the sets $U_{[\gamma]}$ form a topology on $\tilde{X}$. Indeed, given $U_{[\gamma]}, V_{[\gamma']}$, and $[\gamma''] \in U_{[\gamma]} \cap V_{[\gamma']}$, we have that $U_{[\gamma]} = U_{[\gamma'']}$ and $V_{[\gamma']} = V_{[\gamma'']}$. In turn, for any $W \subset U \cap V$, with $\gamma''(1) \in W$, we have that
$$W_{[\gamma'']} \subset U_{[\gamma'']} \cap V_{[\gamma'']}$$
and also $[\gamma''] \in W_{[\gamma'']}$. [Note here that if $U_{[\gamma]} \cap V_{[\gamma']} \neq \emptyset$ then also $U \cap V \neq \emptyset$, and also $\gamma''(1) \in U \cap V$]. This means that the sets $U_{[\gamma]}$ form a topology on $\tilde{X}$.  
Note that the bijection $U_{[\gamma]} \to U$ is a homeomorphism, since it gives a bijection between the subsets $V_{[\gamma']} \subset U_{[\gamma]}$ and sets $V \in \mathcal{U}$ contained in $U$. This also implies that $p$ is continuous. *Construction for choosing basepoints missing.* $\blacksquare$


**Lemma.** In the setting of Proposition 5., $\mathcal{U}$ is a basis for the topology of $X$.

**Proof.** Let $V \subset X$ be open; we'd like to find an open set $U \subset V$ such that $u \in \mathcal{U}$. Pick $y \in V$. Then, since $X$ is locally simply-connected, there is an open $U$ so that $x \in U \subset V$, and $U$ is simply connected, and thus $U \in \mathcal{U}$. $\blacksquare$


Notice that any simply-connected metric space induces a trivial homomorphism into $\pi_1(X, x_0)$. Now we extend this:

**Proposition 6.** Suppose that $X$ is path-connected, locally path-connected, and semilocally simply connected. Then, for every subgroup $H \subset \pi_1(X, x_0)$ there is a covering space $p: X_H \to X$ such that $p_*(\pi_1(X_H, \tilde{x_0})) = H$, for a suitably chosen base-point $\tilde{x_0} \in X_H$.

**Proof.** Let $\tilde{X}$ be the space constructed in Proposition 5. Define the relation $[\gamma] \sim [\gamma']$ when $\gamma(1) = \gamma'(1)$ and $[\gamma \overline{\gamma'}] \in H$. It can be seen that this is an equivalence relation. It can be seen $X_H$, the quotient space of $\tilde{X}$ by the relation $\sim$. Note also that if $\gamma \sim \gamma'$, then $U_{[\gamma]} = U_{[\gamma']}$, thus upon identifying $[\gamma]$ with $[\gamma']$, the whole basic neighborhoods are identified. This means that the propjection $X_H \to X$ induced by $[\gamma] \to \gamma(1)$ is a covering space.  
Choose the basepoint $\tilde{x_0} \in X_H$, to be the equivalence class of the constant path at $c$ at $x_0$. Then, for any loop $\gamma$ in $X$ based at $[x_0]$, its lift to $\tilde{X}$ that starts at $\tilde{x_0}$ ends at $[\gamma]$, which is a loop in $X_H$ iff $[c] \sim [\gamma]$, i.e. $c(1) = \gamma(1)$ and $[\gamma \overline{c}] = [\gamma] \in [H]$. $\blacksquare$

Given the above theorems, we can also ask whether these covering spaces corresponding to different subgroups of the fundamental group are unique.

**Definition.** We say that a homeomorphism $f$ between two covering spaces $p_1: \tilde{X_1} \to X, p_2: \tilde{X_2} \to X$ is an **isomorphism** if $p_1 = p_2 f$.

Note that the inverse of an isomorphism is also an isomorphism, and the composition of isomorphisms is also an isomorphism.

**Proposition 7.** If $X$ is path connected and locally path connected, then two path connected covering spaces $p_1: \tilde{X_1} \to X$ and $p_2: \tilde{X_2} \to X$ are isomorphic *iff* 
$$p_{1*}\left( \pi_1(\tilde{X_1}, \tilde{x_1}) \right) = p_{2*}\left( \pi_1(\tilde{X_2}, \tilde{x_2}) \right)$$

**Proof.** If there is an isomorphism $f: (\tilde{X_1}, \tilde{x_1}) \to (\tilde{X_2}, \tilde{x_2})$, then from $p_1 = p_2 f$ we have $p_{1*} = p_{2*} f_{*}$
$$p_{1*}\left( \pi_1(\tilde{X_1}, \tilde{x_1}) \right) \subseteq p_{2*}\left( \pi_1(\tilde{X_2}, \tilde{x_2}) \right)$$
Since $f^{-1}$ is also an isomorphism, the reverse inclusion also holds, implying one direction of the proposition.  
For the converse, suppose that $p_{1*}\left( \pi_1(\tilde{X_1}, \tilde{x_1}) \right) = p_{2*}\left( \pi_1(\tilde{X_2}, \tilde{x_2}) \right)$. By Proposition 3, $p_1$ has a lift $\tilde{p_1}: (\tilde{X_1}, \tilde{x_1}) \to (\tilde{X_2}, \tilde{x_2})$ so that $p_2 \ \tilde{p_1} = p_1$, and by applying the same reasoning to $p_2$ we obtain $\tilde{p_2}$ so that $p_1 \tilde{p_2} = p_2$. Since $\tilde{p_1}(\tilde{x_1}) = \tilde{x_2}$, and $\tilde{p_2}(\tilde{x_2}) = \tilde{x_1}$, we have that $\tilde{p_1}\tilde{p_2} = \mathbf{1}$ by Proposition 4 (by considering the lift of $\tilde{p_1} p_2$). $\blacksquare$

All our results culminate in the following classification theorem:

**Theorem 1.** Let $X$ be a path-connected, locally path-connected and semilocally simply connected. Then, there is a bijection between the set of basepoint-preserving isomorphism classes of path-connected covering spaces $p:(\tilde{X}, \tilde{x_0}) \to (\tilde{X_1}, \tilde{x_1})$ and the set of subgroups of $\pi_1(X, x_0)$.

Informally speaking, changing basepoints usually comes with prefixing, and post-fixing every loop with a path. 

**Theorem 2.** Consider the setup of Theorem 1. If the basepoints are ignored, the correspondence of Theorem 1 gives a bijection between isomorphism classes of path-connected covering spaces $p: \tilde{X} \to X$ and conjugacy classes of subgroups of $\pi_1(X, x_0)$.  

**Proof.** Suppose that $p:(\tilde{X_0}, \tilde{x_0}) \to (X, x_0)$ is a covering space. We will show that changing the basepoint in $\tilde{X}$ within $p^{-1}(x_0)$ corresponds to changing $p_*\left( \pi(\tilde{X}, \tilde{x_0}) \right)$ to a conjugate subgroup.  
Suppose that $\tilde{x_1} \in p^{-1}(x_0)$, and let $\gamma$ be a path in $\tilde{X}$ from $\tilde{x_0}$ to $\tilde{x_1}$ (which exists since $\tilde{X}$ is path connected). Then $\gamma$ projects to a loop $\gamma$ in $X$, representing some $g \in \pi_1(X, x_0)$. Set $H_i = p_*(\pi_1(\tilde{X}, \tilde{x_i}))$. Clearly $g^{-1} H_0 g \subset H_1$, since for any loop $\tilde{f}$ at $\tilde{x_0}$, conjugating with the path $\tilde{\gamma}$ yields a loop at $\tilde{x_1}$. Similarly we have that $g^{-1}H_1 g \subset H_0$, thus we see that changing the basepoint from $\tilde{x_0}$ to $\tilde{x_1}$ takes $H_0$ to the conjugate subgroup $H_1 = g^{-1} H_0 g$.  
Conversely, if we change $H_0$ to a conjugate subgroup $H_1 = g^{-1} H_0 g$, choose a loop $\gamma$ representing $g$, and lift it to a path $\tilde{\gamma}$ starting at $\tilde{x_0}$ and let $\tilde{x_1} = \tilde{\gamma}(1)$. By repeating the preceding argument, we see that $H_1 = g^{-1} H_0 g$.
