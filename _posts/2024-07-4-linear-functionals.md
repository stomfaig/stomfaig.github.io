---
layout: post
title: "Characters, states and Gelfand-Naimark"
mathjax: true
categories: misc
---

In this article I aim to introduce two ways to represent C\*-algebras.

*Disclaimer: My notes heavily influenced by 'An introduction to the theory of C\*-algebras' by G. J. Murphy, and many proofs and developments are the exact ones in the book. These documents are intended to help for those reading the book, by providing more detail in some cases, or by organizing the material in a different way.*

---

When working with C\*-algebras one has some really strong tools available to tackle certain problems. These arise from *representations*: Ways in which we can identify a C\*-algebra with more familiar objects. There several ways in which this can be done, however.


## Characters and the Gelfand representations.

We start by considering an abelian Banach algebra. As it turns out, in this case we can derive powerful results for a relatively low cost.

**Definition**. A *character* on a Banach algebra $A$ is a homomorphism $\tau: A \to \mathbb{C}$. We denote by $\Omega(A)$ the set of all characters on $A$.

Characters can be thought of as a generalization of the spectrum of an element:

**Theorem 1.** Let $A$ be an abelian Banach algebra. If $A$ is unital,
<center> $$\sigma(a) = \{ \tau(a): \tau \in \Omega(A) \} \quad \forall a \in A$$ </center>
while if $A$ non-unital, then
<center> $\sigma(a) = \{0 \} \cup \{\tau(a): \tau \in \Omega(A) \} \quad \forall a \in A$ </center>

**Proof**. We start with the unital case. Given $a \in A$, and $\lambda \in \sigma(a)$, consider $I = (a- \lambda) A$, which is a proper ideal (...). Since $A$ is unital, there is a maximal ideal $J$ that contains $I$ (since $I$ modular as $A$ unital, and modular ideals are contained in maximal ideals). Since the maximal ideals of an abelian Banach algebra are in one-to-one correspondence with kernels of characters (see Murphy Theorem 1.3.3.), there is $\tau \in \Omega(A)$ such that $\ker \tau = J$. This way, $\tau((a - \lambda) A ) = 0$, so $\tau((a- \lambda) \mathbf1) = 0$, and hence $\tau(a) = \lambda$.  
In the non-unital case, the characters of $\tilde{A}$ are the unique extensions of characters of $A$, and the canonical embedding $\tau\_\infty: \tilde{A} \to \mathbb{C}$. This implies the result.  $\blacksquare$  

**Remark.** From the discussion this far it is  not obvious why one needs the criterion that $A$ is abelian. The restriction is necessary, so that the underlying algebra works out: the correspondence between maximal ideals and kernels of characters heavily relies on the fact that $A$ is abelian. Even though this is the case, I don't think that the associated algebra is any enlightening.

We define the *Gelfand representation* $\widehat{a}$ of an element $a \in A$ as follows:
<center> $\widehat{a} : \Omega(A) \to \mathbb{C} \quad \widehat{a}(\tau) = \tau(a)$ </center>
Notice that the function $\widehat{a}$ is continuous of compact support. Indeed, if $\tau_n \to \tau$, then $\tau_n(a) \to \tau(a)$, since $\Omega(A)$ has the weak-* topology, and therefore $\widehat{a}(\tau_n) \to \widehat{a}(\tau)$. Further, given $\epsilon > 0$ and consider the set $\\\{ |\widehat{a}(\tau)| > \epsilon : \tau \in \Omega(A) \\\}$ is a weak-\* closed set in the unit ball of $A^\*$, therefore it is weak-\* compact. This means that $\widehat{a} \in C_0(\Omega(A))$.

Now we are ready for the definitive result of Gelfand representations.

**Theorem 2.** The map $\widehat{\cdot}: A \to C_0(\Omega(A))$ is a norm-decreasing homomorphism, with $\\\| \widehat{a}\\\|_\infty = r(a)$.

**Proof.** To prove that $\widehat{\cdot}$ is a \*-homomorphism, we just need to use the definition of $\widehat{a}$ and the fact that all $\tau \in \Omega(A)$ are themselves \*-homomorphisms:
<center> $$\widehat{a + b}(\tau) = \tau(a+b) = \tau(a) + \tau(b) = \widehat{a}(\tau) + \widehat{b}(\tau)$$ </center>
<center> $$\widehat{ab}(\tau) = \tau(ab) = \tau(a)\tau(b) = \widehat{a}(\tau)\widehat{b}(\tau)$$ </center>
<center> $$\widehat{a^*}(\tau) = \tau(a^*) = \tau(a)^* = \widehat{a}^*(\tau) $$</center>
Now, by Theorem 1., $\sigma(a) \cup \\\{0 \\\} = \widehat{a}(\Omega(A))$, from which we obtain that $\\\| \widehat{a} \\\|\_\infty = r(a) \leq \\\| a\\\|$. $\blacksquare$

In the case when $A$ is a C\*-algebra, we can say even more. Indeed, suppose that $\phi$ is \*-homomorphism between C\*-algebras. Then, using the C* identity $\\\| a \\\|^2 = \\\| a^* a \\\|$ in $\mathbb{C}$,  
<center>$$\| \phi(a)\|^2=\|\phi(a)^* \phi(a) \| = \| \phi(a^* a ) \| = r(a^*a)$$ </center>
Since $a^\* a$ is self-adjoint, we have that $ r(a^\*a) = \\\| a^\* a \\\| $, and therefore $\\\| \phi(a) \\\| = \\\| a \\\| $. One can easily prove that the Gelfand representation is itself a \*-homomorphism, and therefore an isometry. Note further, that $\\widehat{\cdot} : A \to C_0(\Omega(A))$ is an isomorphism, since the image $\widehat{A}$ is dense in $C_0(\Omega(A))$ by the Stone-Weierstrass theorem. Combining all these results, we obtain the following

**Theorem 3.** Let $A$ be an abelian C\*-algebra. Then, the Gelfand representation $\widehat{\cdot} : A \to C_0(\Omega)$ is an isometric \*-isomorphism.

## Linear functionals and States.

**Definition.** A a linear map $\varphi: A \to B$ between C\*-algebras is *positive*, if $\varphi(A^+) \subseteq B^+$. 

Suppose that $\varphi: A \to \mathbb{R}$ is a positive linear functional. Then, the map  
<center> $$A^2 \to \mathbb{C}, \quad (a,b) \to \varphi(b^*a)$$ </center> 
is a sesquilinear form, further $(a,a) = \varphi(a^\*a) \in \mathbb{R}^+$ (since $a^\*a \in A^+ \ \forall a \in A$). We also note that the Cauchy-Swartz inequality holds for $(\cdot, \cdot)$, by the usual proof.

The following lemma is really useful for determining whether a linear function is positive or not.

**Lemma.** Suppose that $\varphi$ is a bounded linear functional on the C\*-algebra $A$, and let  be an  in $A$. Then, the following are equivalent.
1. $\varphi$ is positive,
2. For any approximate unit $(u\_\lambda)\_{\lambda \in \Lambda} $ in $A$, $\lim\_{\lambda} \varphi(u\_\lambda) = \\\| u \\\|$,
3. For some approximate unit $(u\_\lambda)\_{\lambda \in \Lambda} $ in $A$, $\lim\_{\lambda} \varphi(u\_\lambda) = \\\| u \\\|$.

**Proof.** Without loss of generality we may assume that $\\\| \varphi \\\| = 1$.  
$(1) \implies (2)$: Suppose that (1) holds, and let $(u\_\lambda)\_{\lambda \in \Lambda}$ be an approximate unit for $A$. Then, clearly $\lim\_\lambda \varphi(u\_\lambda) \leq 1$. Now, let $a \in \lambda$ with $\\\| a \\\| \leq 1$. Then,  
$$| \varphi(u_\lambda a)|^2 \leq \varphi(u_\lambda ^2) \varphi(a^* a) \leq \varphi(u_\lambda) \leq \lim_\lambda \varphi(u_\lambda)$$  
Since $(\varphi(u\_\lambda))\_\lambda$ is an increasing sequence in $\mathbb{R}$, the limit on the right-hand side is finite.  
$(2) \implies (3)$ is clearly true.  
$(3) \implies (1)$ *Coming soon.*

**Corollary.** A linear functional $\varphi$ on a C\*-algebra $A$ is positive *iff* $\\\| \varphi \\\| = \| \varphi(1) \|$

**Definition.** By a *state* on a C\*-algebra we mean a positive linear functional $\varphi$, so that $\\\| \varphi \\\| = 1$.

**Proposition.** Let $A$ be a unital C\*-algebra, and $a \in A$ normal. Then, there is a state $\varphi$ on $A$ so that $\\\| a \\\| = \varphi(a)$.

**Proof.** Consider the C\*-algebra $B$ generated by $\mathbf1$ and $a$. Since $a$ normal, $B$ is abelian, thus by Gelfand representation $\\\| \widehat{a} \\\|\_\infty = \\\| a \\\|$, where $\widehat{a}: \Omega(B) \to \mathbb{C}$ is a continuous function, and $\Omega(B)$ is compact. Therefore we can find a character $\tau \in \Omega(A)$ so that $\tau(a) = \\\| a \\\|$, and we know that $\tau(\mathbf1) = 1 = \|\tau\|$, and $\tau$ is clearly linear and bounded, so by the preceding Corollary $\tau$ is a state on $B$.  
By the Hahn-Banach theorem we might extend $\tau$ to $A$, while preserving its norm, to say $\tau'$. Now, $\tau'$ is clearly linear and bounded, further $ \tau'(\mathbf1) = \tau(\mathbf1) = \\\| \tau \\\| = \\\| \tau' \\\|$, therefore $\tau'$ is positive. Clearly $\tau(a) = \\\| a \\\|$.

**Remark.** This proof goes through even in the non-unital case, by considering the unitisation of $A$ and $B$.

## Representations.

**Definition.** A *representation* of a C\*-algebra $A$ is a pair $(H, \varphi)$ where $H$ is a Hilbert space, and $\varphi: A \to H$ is a \*-homomorphism. We say that a representation $(H, \varphi)$ is *faithful*, if $\varphi$ is injective.

We can also combine representations: Suppose that $(H\_\lambda, \varphi\_\lambda)\_{\lambda \in \Lambda}$ is a family of representations of $A$, then their *direct sum* $(H, \varphi)$ is given by
<center> $$H = \oplus_\lambda H_\lambda, \quad \varphi(a)(x_\lambda)_\lambda = (\varphi_\lambda(a)(x_\lambda))_\lambda $$ </center>
This is helpful to obtain faithful representations: given $x \in H$, if $\varphi\_\lambda(x) \neq 0$ for some $\lambda \in \Lambda$, then $\varphi(a) \neq 0$. This way, if for each $x \in H$, there is a representation not vanishing on $x$, their direct sum is faithful.  

Given a positive linear function $\tau$ on $A$, we construct an associated representation. First, consider $N_\tau = \\\{ a \in A : \tau(a^\* a) = 0 \\\}$.

**Lemma.** Given a positive linear functional $\tau$, $N\_\tau$ is a closed left-ideal.

**Proof.** *Easy exercise.* $\blacksquare$

The map
<center> $$(A/N_\tau)^2 \to \mathbb{C}, \quad (a+ N_\tau, b + N_\tau) \to \tau(b^*a)$$ </center>
is easily verified to be an inner product on $A/N\_\tau$, which makes it a pre-inner product space. Let $H\_\tau$ be the completion of $A/N\_\tau$, and define the operator $\varphi(a) \in B(A / N_\tau)$ as 
<center> $$ \varphi(a)(b + N_\tau) = ab + N_\tau $$ </center>
The operator $\varphi(a)$ has a unique extension to $H\_\tau$, which we call $\varphi_\tau$. The map
<center> $$ \varphi_\tau: A \to B(H_\tau), \quad a \mapsto \varphi_\tau(a)$$ </center>
is a \*-homomorphism.  
What we obtained this way, is the GNS (Gelfand-Naimark-Segal) representation associated to $\tau$. We call the direct sum of all the representations associated to states of $A$ the *universal representation* of $A$. This turns out to be faithful.

**Theorem.** The universal representation is faithful.

**Proof.** Given $a \in A$, $a^\*a$ is normal, hence by Proposition ?? there is a state $\tau$ on $A$ such that $\tau(a^*a) = \\\| a^\* a \\\|$. Consider $b = (a^\* a) ^{1/4}$. Then,
<center> $$\| a \|^2 = \| a^* a \| = \tau(a^* a) = \tau(b^4) = \| \varphi_\tau(b)(b + N_\tau) \|^2 \quad (\dagger)$$ </center>
Where the last equality follows from: $\\\| \varphi\_\tau(b)(b + N_\tau) \\\|^2 = \\\| b^2 + N \\\|^2 = \tau((b^2)^\* b^2) = \tau(b^4)$ since $b$ is self-adjoint.  
Let $\varphi$ be the universal representation of $A$. If $\varphi(a) = 0$, then $\varphi\_\tau(a) = 0$, and $\varphi_\tau(b^4) = \varphi(a^\* a) = \varphi(a^\*) \varphi(a)$ (even though $\tau$ is not a homomorphism, $\varphi\_\tau$ is). Thus, $\varphi(b) = 0$ as well. Substituting into $(\dagger)$ yields that $\\\| a \\\|  =0$, thus $\varphi$ is faithful. $\blacksquare$

---

*Latest revision: 4th of July, 2024*
