---
layout: post
title: "Covering spaces and permutations."
mathjax: true
categories: misc
---


*This post is an adoptation of Chapter 1.3 of Allan Hatcher's Algebraic Topology. As such, it contains no original work apart from the style of the exposition, and likely errors.*

---

In a previous article we discussed a way of thinking about path-connected covering spaces of path-connected, locally path-connected, semilocally simply-connected spaces as subgroups of the fundamental group of the original space. In this article we present a different way of classifying covering spaces.  

Suppose that $X$ is path-connected, locally path-connected and semilocally simply-connected, and let $p: \tilde{X} \to X$ be a covering space for $X$. Given a path $\gamma$, this has a unique lift $\tilde{\gamma}$ with starting point $\tilde{x} \in p^{-1}(\gamma(0))$. Note also that $\tilde{\gamma}(1) \in p^{-1}(\gamma(1))$, thus we obtain a well-defined map
$$L_\gamma : p^{-1}(\gamma(0)) \to p^{-1}(\gamma(1)), \quad \tilde{\gamma}(0) \mapsto \tilde{\gamma}(1)$$
Clearly $L_\gamma$ is a bijection with inverse $L_{\overline{\gamma}}$, for a composition of paths $\gamma \eta$ we have $L_{\gamma \eta} = L_\eta L_\gamma$.  
This is somewhat annoying, but upon rather setting $L_\gamma$ to be its inverse, we have that $L_{\gamma \eta} = L_\gamma L_\eta$.  
Pick $x_0 \in X$, and consider the mapping $\gamma \mapsto L_\gamma$ for $\gamma \in \pi_1(X, x_0)$. Clearly $L_\gamma$ only depends on the homotopy class of $\gamma$ (since any homotopy can be lifted to $\tilde{X}$ to give a homotopy), so this mapping is wel-defined. All the $L_\gamma$ are permutations of the fiber $p^{-1}(x_0)$. We call this the *action of $\pi_1(X, x_0)$ on the fiber $p^{-1}(x_0)$*.

**Theorem 1.** Suppose that $X$ is a path-connected, locally path-connected and semilocally simply-connected topological space. A covering space $p: \tilde{X} \to X$ can be reconstructed from the associated action of $\pi_1(X, x_0)$ on the fiber $F= p^{-1}(x_0)$.

**Proof.** Let $\rho$ be the action of $\pi(X, x_0)$ on the fiber $p^{-1}(x_0)$.  
By Proposition 5. from the the article '', $X$ has a universal cover $\tilde{X_0} \to X$, which we can think of as the space of homotopy classes of paths in $X$.  
Define a map $h: \tilde{X_0} \times F \to \tilde{X}$ sending a pair $([\gamma], \tilde{x_0})$ to $\tilde{\gamma}(1)$, where $\tilde{\gamma}$ is the unique lift of $\gamma$ to $\tilde{x_0} \in F$.  
continuity...surjectivity...  
Now we investigate the kernel of $h$. 
Suppose now that $h([\gamma], \tilde{x_0}) = h([\gamma'], \tilde{x_0}')$. Then $\gamma$ and $\gamma'$ are both paths from $x_0$ to the same endpoint, and $\tilde{\gamma}(1) = \tilde{\gamma'}(1)$ by the definition of $h$. This implies, that $L_{\gamma' \overline{\gamma}}(\tilde{x_0}) = \tilde{x_1}$. Set $\lambda = \gamma' \overline{\gamma}$. Then:
$$h([\gamma], \tilde{x_0}) = h([\lambda \gamma], \rho(\lambda)(\tilde{x_0}'))$$

Conversely, for any loop $\lambda$, we have that $h([\gamma], \tilde{x_0}) = h([\lambda \gamma], \rho(\lambda)(\tilde{x_0}))$, which means that $h$ induces a well defined map to $\tilde{X}$ from the quotient space of $\tilde{X_0} \times F$ with respect to the relation $([\gamma], \tilde{x_0}) \sim ([\lambda \gamma], \rho(\lambda)(\tilde{x_0}))$ for each $[\lambda] \in \pi_1(X, x_0)$. Denote this quotient space by $\tilde{X_\rho}$.  
The natural projection $([\gamma], \tilde{x_0}) \mapsto \gamma(1)$ induces a covering space, since for any open set $U$ so that $\tilde{X_0}$ is a disjoint union of sets homeomorphic to $U$, then since $\tilde{X_0}$ is path connected, the identification defining $X_\rho$ collapses all these sheets to $U \times F$.  
All that's left to do is to prove that $X_\rho$ is isomorphic to $\tilde{X}$. We've seen before that $h$ is surjective, and after quotienting $h: \tilde{X_\rho}$ is both surjective and injective. Since $h$ is a local homeomorphism, it is also a homeomorphism. Note also, that this map takes each fiber of $\tilde{X_\rho}$ to the corresponding fiber of $\tilde{X}$. This implies that $\tilde{X_\rho}$ is isomorphic to $\tilde{X}$. $\blacksquare$

Let $p: \tilde{X} \to X$ be a covering space. An isomorphism $\tilde{X} \to \tilde{X}$ is called a **deck transformation**. Clearly the deck transformations over $\tilde{X}$ form a group under composition, which we will call $G(\tilde{X})$. We note that, if $\tilde{X}$ is path connected, then by the unique lifting property any deck transformation is determined by the image of a single point.   
We say that a covering space $p: \tilde{X} \to X$ is normal, if for every $x \in X$, and each pair $\tilde{x_1}, \tilde{x_2}$ of lifts of $x$ to $\tilde{X}$ there is a deck transformation taking $\tilde{x_1}$ to $\tilde{x_2}$.

**Proposition 2.** Let $p: (\tilde{X}, \tilde{x_0}) \to (X, x_0)$ be a path-connected covering space of the path-connected, locally path-connected space $X$, and let $H = p_*(\pi_1(\tilde{X}, \tilde{x_0})) \subset \pi_1(X, x_0) = G$. Then,  
1.  $p: (\tilde{X}, \tilde{x_0}) \to (X, x_0)$ is normal *iff* $H$ is normal in $G$,
2. $G(\tilde{X})$ is isomorphic to the quotient $N(H)/N$ where $N(H)$ is the normalizer of $H$ in $G$.

**Proof.** 1. Suppose that $[\gamma] \in G$, and let $\tilde{\gamma}$ be the lift of $\gamma$ at the point $\tilde{x_0}$, and call $\tilde{\gamma}(1) = \tilde{x_1}$. Then, by Theorem 2. of the previous article $\gamma \in N(H)$ *iff* $p_*(\pi_1(\tilde{X}, \tilde{x_0})) = p_*(\pi_1(\tilde{X}, \tilde{x_1}))$. This, by the lifting criterion implies that there is a deck transformation taking $\tilde{x_0}$ to $\tilde{x_1}$. Note also that for any $\tilde{x_0}, \tilde{x_1}$ in $p^{-1}(x)$ there is a path $\tilde{\gamma}$ in $\tilde{X}$ connecting these points, since $\tilde{X}$ is path-connected.  This also gives a loop $\gamma = p(\tilde{\gamma})$ based at $x_0$, which lifts to $\tilde{\gamma}$. From this we conclude that $H$ normal *iff* $N(H) = G$ *iff* the covering space is normal.  
2. Define $\psi : N(H) \to G(\tilde{X})$ by sending $[\gamma]$ to the deck transformation $\tau$, that $\gamma$ induces as above. Clearly $\psi$ is a group homomorphism, and is clearly surjective based on the above paragraph. Note further, that the kernel of $\psi$ consists of loops in $(X, x)$ that lift to loops in $\tilde{X}$, which are exactly the elements of $p_*(\pi_1(\tilde{X}, \tilde{x_0}))$ by Proposition 2 from the previous article. $\blacksquare$

We can generalize the notion of deck transformation. At its core, the group of deck transfomation is obtained by injecting a group into the homeomorphism group of a space. Thus we define:

**Definition.** Given a group $G$ and a space $Y$, then an **action** of $G$ on $Y$ is a homomorphism $\rho$ from $G$ to $\text{Homeo}(Y)$, the group of all homeomorphism from $Y$ to itself.

We will usually use the notation $g: Y \to Y$ to refer to $\rho(g): Y \to Y$, when it is clear which action we're referring to.

**Definition.** We call an action that satisfies  
> Each $y \in Y$ has a neighborhood $U$ such that all the images $g(U)$ are all disjoint for different $g \in G$.  

a **covering space action**.

**Proposition 3.** The action of the deck trasnforamtion group $G(\tilde{X})$ on $\tilde{X}$ is a covering space action.

**Proof.** *Coming soon.* $\blacksquare$


Given the action of a group $G$ on space $Y$, we can form the quotient space $G/Y$ in which each $y \in X$ is identified with all images $g(y)$ for $g \in G$. We call the points obtained this way **orbits** $Gy = \{g(y) : g \in G\}$, and we call $G/Y$ the **orbit space** of the action.

**Proposition 4.** Given a covering space action $G$ on a space $Y$, we have  
1. The quutient map $p: Y \to Y/G, p(y) = Gy$ is a normal covering space,
2. if $Y$ is path connected, $G$ is the group of deck-transformations for $G/Y$,
3. If $Y$ is path-connected and locally path-connected, then $G$ is isomorphic to $\pi_1(Y/G) / p_*(\pi_1(Y))$.

**Proof.** 1. We start by proving that $Y$ is a covering space. Since $G$ is a covering space action, $p$ identifies all the disjoint homeomorphic sets $\{g(U) : g \in G\}$. By the definition of the quotient topology $p$ is a homeomorphism pn all these sheets, thus we have a covering space.  
It is clear that each element of $G$ acts as a deck transformation, further since $G$ is a group, it is transitive for all the orbits.  
If $Y$ is path connected and $f$ is a deck transformation, then for any $y \in Y$, we have $y$ and $f(y)$ in the same orbit, thus there is $g \in G$ with $g(y) = f(y)$. Since deck transformations on path-connected spaces are determined by the image of a single point, we must have that $f = g$.
