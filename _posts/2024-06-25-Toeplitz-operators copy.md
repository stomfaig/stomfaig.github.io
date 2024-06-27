---
layout: post
title: "Toeplitz operators"
mathjax: true
categories: misc
---

*Disclaimer: My notes heavily influenced by 'An introduction to the theory of C\*-algebras' by G. J. Murphy, and many proofs and developments are the exact ones in the book. These documents are intended to help for those reading the book, by providing more detail in some cases, or by organizing the material in a different way.*

*This article is not complete yet*

---

Consider the circle group $\mathbb{T}$ endowed with the arc length measure. Define for each $n \in \mathbb{Z}$ the function $\varepsilon_n: \mathbb{T} \to \mathbb{T}$ given by $\lambda \to \lambda^n$. We denote by $\Gamma$ the closed linear span of the $\varepsilon_n$.

**Theorem 1.** $\Gamma$ is norm dense \*-subalgebra of $C(\mathbb{T})$.

**Proof.**  $$\tag*{\(\blacksquare\)}$$

Since $C(\mathbb{T})$ is dense in $L^p(\mathbb{T})$ for every $p$, we have the following.

**Corollary.** $\Gamma$ is norm dense in the spaces $L^p(\mathbb{T})$.

In $L^2(\mathbb{T})$, $\int \varepsilon_n(\lambda) \overline{\varepsilon_m(\lambda)} d \lambda = \int_{0}^{2\pi} e^{i n \theta } e^{im \theta} d \theta = C \cdot \delta_{n,m}$ for some constant, hence $\\\{ \varepsilon_n \\\}\_{n \in \mathbb{Z}}$ is an orthonormal basis for the Hilbert space $H^2(\mathbb{T})$. Given $f \in L^1(\mathbb{T})$, we define the Fourier transform $\widehat{f}$ of $f$ as  
<center> $$\widehat{f}(n) = \int f(\lambda) \overline{\varepsilon_n(\lambda)} d \lambda$$ </center> 


**Definition.** For $p \in [1, \infty]$, we define the *Hardy space*  
<center>$$H^p = \{f \in L^p(\mathbb{T}) : \widehat{f}(n) = 0, \ \forall n < 0\}$$ </center>

Note that $\\\{ \varepsilon_n\\\}\_{n \in \mathbb{N}}$ is an orthonormal basis for $H^p$.  

In what follows, we will mainly work in $L^2(\mathbb{T})$ and in $H^2$. We do this for the natural reason: these are Hilbert spaces under the inner-product $(f, g) = \int f \bar{g}$, and therefore the algebra of their operators is a C\*-algebra.

**Definition.** Suppose that $\varphi \in L^\infty(\mathbb{T})$. Then, we define the multiplication operator $M_\varphi$ as follows:
<center> $$M_\varphi \in B(L^2(\mathbb{T})), \quad M_\varphi(f) = \varphi f$$ </center>

And note that the map $L^\infty(\mathbb{T}) \to B(L(\mathbb{T})), \varphi \mapsto M_\varphi$ is an isometric *-homomorphism.

Consider the operator $u = M\_{\varepsilon_1}$. This is the unilateral shift operator in $L^2(\mathbb{T})$ the basis $\\\{ \varepsilon_n \\\}\_{n \in \mathbb{Z}}$, and the left-shift in $H^2$ with respect to the basis $\\\{ \varepsilon_n \\\}\_{n \in \mathbb{N}}$.

**Theorem 2.** If $w \in B(L^2(\mathbb{T}))$, then $w$ commutes with $v$ *iff* $w = M\_\varphi$ for some $\varphi \in L^\infty (\mathbb{T})$.

**Remark.** This result can be made much more familiar, upon putting it into the right perspective. 

**Proof.** $(\implies)$ Consider $\psi \in \Gamma$. Then, $M_\psi$ is in the span of the operators $\\\{ v^n \\\}_{n \in N}$, as the map $\psi \mapsto M\_\psi$ is a homomorphism, as noted before. Then, since $w$ commutes with $v$, it also commutes with all the $M\_\psi$'s, for $\psi \in \Gamma$.  
Since $\Gamma$ is norm-dense in $L^2(\mathbb{T})$, for any $\psi \in L^2(\mathbb{T})$, there is a sequence $\psi_n \in \Gamma$ so that $\psi_n \to \psi$ in $L^2$, and so $w(\psi_n) \to w(\psi)$ in $L^2$. Upon passing to a subsequence we may assume that $w(\psi_n) \to w(\psi)$ almost everywhere.  
Define $\varphi = w(\varepsilon_0)$. Then, for each $n$, $\psi_n = \psi\_n \varepsilon\_0 = M\_{\psi\_n} \varepsilon\_0$. Since $\psi\_n \in \Gamma$, $\psi\_n$ and $w$ commute $\forall n$, so
<center> $$ w(\psi_n) = w M_{\psi_n} (\varepsilon_0) = M_{\psi_n} w(\varepsilon_0) = \psi_n \varphi$$ </center>
Now, by considering the sets $E\_n = \\\{ \lambda \in \mathbb{T} : |\varphi(\lambda)| > \| w\| + 1/n \\\}$, it follows that $|\varphi(\lambda) | < \| w \|$.  
$(\impliedby)$ Suppose that $w = M\_\varphi$ for some $\varphi \in L^\infty(\mathbb{T})$. Then,
<center> $$v M_\varphi (f) = v ( \varphi f) = \varphi v(f) = M_\varphi(vf) \tag*{\(\blacksquare\)}$$ </center>

If $E$ is Borel set of $\mathbb{T}$, then $M\_{\chi\_E}$ is a projection on $L^2(\mathbb{T})$, which is easily checked.

**Definition.** Let $E$ be as above. We call the range of $M\_{\chi\_E}$ is called a *Wiener subspace* of $L^2(\mathbb{T})$.

Note that for any Wiener subspace $K$, we have that $v(K) =K$. This follows from the fact $v$ and $M\_{\chi\_E}$ commute as per Theorem 2.  

**Definition.** Let $\varphi \in L^\infty(\mathbb{T})$ be unitary. Then $\varphi H^2$ is a closed subspace of $L^2(\mathbb{T})$, called a *Beurling space*.

**Theorem 3.** The invariant subspaces of $L^2(\mathbb{T})$ under $v$ are exactly the Wiener and Beurling spaces. If $K$ is an invariant subspace for $v$, further,
1. if $v(K) = K$, then $K$ is a Wiener subspace,
2. if $v(K) \neq K$, then $K$ is a Beurling subspace.

**Proof.** Suppose that $v(K) = K$, and let $p \in B(L^2(\mathbb{T}))$ be the projection operator onto the subspace $K$. Then, $pv$ and $vp$ clearly agree on $K$. Further, if $f \in K^\perp$, notice that $vp(f) = v(0) = 0$. Further, $v^\*$ is the left-shift, which is the inverse of $v$, so $v^\*(K) = K$ as well. From this,
<center> $$\langle v(f), g \rangle = \langle f, v^*(g) \rangle = 0 \quad \forall g \in K$$ </center>
Therefore $vp = pv$. By Theorem 2, there is $\varphi \in L^\infty(\mathbb{T})$ so that $p = M\_\varphi$, and so $K = \text{im} \ M\_\varphi$. Since $p$ is projection, and $\varphi \mapsto M\_\varphi$ is a \*-homomorphism, we $\varphi$ is also a projection, i.e. $\varphi = \chi_E$ for some measurable set $E$. From all this it follows that $K$ is a Wiener subspace.  
Now suppose that $v(K) \neq K$. Then, there is a unit vector $\varphi \in K$ such that $\varphi \notin v(K)$. Now, clearly $v^n(\varphi) \subset v(K)$, therefore $\langle v^n(\varphi), \varphi \rangle = 0$. From this,
<center> $$0 = \int \varepsilon_n(\lambda) | \varphi(\lambda)|^2 d \lambda \quad \forall n \in \mathbb{Z}$$ </center>
And from this we have that $|\varphi(\lambda)|^2 = \alpha $ a.e. By the choice of $\varphi$, we have that $\\\| \varphi\\\|_2 = 1$, we must have $\| \varphi(\lambda)|^2 = 1$. We conclude that $\varphi$ is unitary in $L^2(\mathbb{T})$.  
Note that $\varphi H^2 \subset K$, since $H^2$ is the closure of $\Gamma^+$, i.e. for any $\psi \in H^2$, $M\_\psi$ is in the span of $\\\{ v^i \\\}\_{i\geq 1}$, and as $v(K) \subset K$, we have have that $\varphi H^2 \subset K$. $\\\{ \varepsilon_n \varphi \\\}\_{n \in \mathbb{Z}}$ is an orthogonal system, which spans $L^2(\mathbb{T})$. Note also, that for $n < 0$, $\varepsilon\_n \varphi \in K^\perp$, since 
<center> $$\langle \varepsilon_n \varphi, \psi \rangle = \langle \varphi, \varepsilon_{-n} \psi \rangle = 0$$</center>
which follows from the argument before. From these observations it follows that $\\\{ \varepsilon_n \varphi \\\}\_{n \in \mathbb{N}}$ is an orthonormal basis. $$\tag*{\(\blacksquare\)}$$

Now let's look at two interesting applications of this theorem.

**Theorem**(F. and M. Riesz)**.** If $f$ is a function in $H^2$ that does not vanish a.e., then it it vanishes on a set of measure zero.

**Proof.** Let $E = f^{-1}(0)$. Then, consider the norm-closed subspace $K$ of $H^2$, for which $g \in K$ iff $\chi\_E g = 0$ a.e. Then, $v(g) = \varepsilon\_1 g \in H^2$, and $ \varepsilon\_1 g \chi\_E = 0$ a.e., so $v(K) \subset K$. Note also, that $\cap\_n v^n(K) \subset \cap\_n v^n(H^2) = 0$, thus either $v(K) = K = 0$, or $v(K) \neq K$. The former can be easily seen to be absurd. For the latter, by Theorem 3, there is $\varphi \in $ unitary such that $K = \varphi H^2$. Then, $\varphi \chi_E = 0$ a.e., from which $E$ must have measure zero.

**Theorem.** The only closed subspaces of $H^2$ reducing for the unilateral shift are the trivially invariant subspaces.

In what follows, given an innerproduct space $H$, and a closed subsapce $A$, we denote by $H \ominus A$ the complementary subspace of $A$ in $H$.

**Proof.** Suppose that $K$ is a non-trivial subspace of $H^2$ reducing $u$. Then, by arguing as in the proof of the previous theorem, we see that $u(K) \neq K$, so $K$ is a Beurling subspace. Note that $H^2 \ominus K$ is also invariant, and arguing just as before it is also Beurling. Thus, there is $\varphi, \psi \in L^\infty(\mathbb{T})$ unitary, so that $K = \varphi H^2$ and $H^2 \ominus K = \psi H^2$.  
Since $K$ is invariant under $u$, we have that 
<center>  $$\langle u^n \varphi, \psi \rangle = \langle \varepsilon_n \varphi, \psi \rangle = 0 = \langle \varphi, \varepsilon_n \psi \rangle = \langle \varphi, u^n(\psi) \rangle $$ </center>
and therefore $\varphi \overline{\psi}$ has zero Fourier transform, and therefore $\varphi \overline{\psi} = 0$ a.e. Since both of these functions are unitary, this is absurd.


## Toeplitz operators

**Definition.** Let $p$ be the projection of $L^2(\mathbb{T})$ onto $H^2$. If $\varphi \in L^\infty(\mathbb{T})$, we define the *Toeplitz operator* $T\_\varphi$ with symbol $\varphi$:
<center> $$ T_\varphi: H^2 \to H^2, \quad f \mapsto p(\varphi f) $$ </center>
Clearly $\\\| T\_\varphi \\\| \leq \\\| \varphi \|||$. Further, if $\psi \in L^\infty(\mathbb{T})$,
<center>$$T_{\varphi + \psi}(f) = p \big( (\varphi + \psi) f \big) = p(\varphi f) + p( \psi f) = T_{\varphi}(f) + T_{\psi}(f)$$ </center>
hence $\varphi \mapsto T\_\varphi$ is linear. Let $f,g \in H^2$, and write:
<center>$$ \langle T_{\varphi}^*(f), g \rangle = \langle f, T_\varphi(g) \rangle = \langle f, p(\varphi(g)) \rangle = \langle p(f), \varphi(g) \rangle $$ </center>
Now we utilize that $f, g \in H^2$, and so $p(f) = f$ and $p(g) =g$:
<center>$$ = \langle \overline{\varphi}f, p(g) \rangle = \langle p(\overline{\varphi}f), g \rangle $$ </center>
Thus $T\_\varphi^\* = T\_{\overline{\varphi}}$, i.e. $\varphi \to T\_\varphi$ also preserves adjoints.

In general, however, we do not have that $T\_{\varphi \psi} = T\_\varphi T\_\psi$. The following is therefore useful.


**Theorem.** Let $\varphi \in L^\infty$ and $\psi \in H^\infty$. Then,
<center> $$ T_{\varphi \psi} = T_\varphi T_\psi, \quad T_{\overline{\psi} \varphi} = T_{\overline{\psi}} T_\varphi $$ </center>

**Proof.** Since $\psi \in H^\infty$, $\psi H^2 \subset H^2$, therefore $T\_\varphi T\_\psi (f) = p(\varphi p( \psi f)) = p (\varphi \psi f) = T\_{\varphi \psi} (f)$. For the second identity, write $T\_{\overline{\psi} \varphi}^\* = T\_{\overline{\varphi} \psi} = T\_{\overline{\varphi}} T\_{\psi}$, thus taking adjoints gives the result.

### Spectral properties of Toeplitz operators

**Theorem.** Let $\varphi \in L^\infty(\mathbb{T})$, and let $\sigma(\varphi)$ denote the spectrum of $\varphi$ in $L^\infty(\mathbb{T})$. Then, $\sigma(\varphi) \subset \sigma(T\_\varphi)$.

**Remark.** Note $T\_\psi - \lambda = T\_{\psi - \lambda}$, so to prove this theorem, it is sufficient to prove that if $T\_\psi$ is invertible, then so is $M\_\psi$; as the map $\psi \mapsto M\_\psi$ is an isometric \*-homomorphism.

**Proof.** One way to prove that $M\_\psi$ is invertible, is to prove that $M\_\psi^\* M\_\psi$ is *strictly* positive (since $M\_\psi$ is normal).
Since $T\_\psi$ is invertible, it is bounded below, so, for $f \in $, there is a constant $C > 0$ such that  $T\_\psi(f) \geq C \\\| f \\\|$. Note that for any $n \in \mathbb{Z}$, $\\\| M\_\psi( \varepsilon\_n f ) \\\| = \\\| \psi \varepsilon\_n f \\\| = \\\| \psi f \\\| \geq \\\| T\_\psi f \\\| \geq C \\\| f \\\| = C \\\| \varepsilon\_n f \\\|$, so $M\_\psi$ is bounded below on $L^\infty(\mathbb{T})$, as $\\\{ \varepsilon\_n f \\\}\_{n \in \mathbb{Z}}$ is dense in $L^2(\mathbb{T})$. Now, write
<center> finish this off! </center>


## nwe title?

**Definition.** Let $\mathbf{A}$ denote the C\*-algebra generated by all Toeplitz operators $T\_\varphi$ with continuous symbol $\varphi$, called the *Toeplitz algebra*.

**Lemma.**

**Theorem.** The commutator ideal of $\mathbf{A}$ is $K(H^2)$.

**Theorem.** The map 
<center> $$\psi : C(\mathbf{T}) \to \mathbf{A} / K(H^2), \varphi \mapsto T_\varphi + K(H^2)$$ </center>
is a \*-isomorphism.

Recall the following theorem of Atkinson:

**Theorem.** Let X be an infinite-dimensional Banach space and let $u \in B(X)$. Then $u$ is Fredholm *iff* $u + K(X)$ is invertible in the quotient algebra $B(X)/K(X)$.

Combining these two, we have the following corollaries.

**Corollary.** If $\varphi \in C(\mathbb{T})$, then $T\_\varphi$ is Fredholm, *iff* $\varphi$ is nowhere vanishing.

**Proof.** We have that $T\_\varphi$ is Fredholm iff $T\_\varphi + K(H^2)$ is invertible in the quotient $B(H^2)/K(H^2)$, which is equivalent to $T\_\varphi + K(H^2)$ being invertible in $\mathbf{A} / K(H^2)$. Since $\varphi \mapsto T\_\varphi + K(H^2)$ is a \*-isomorphism, this implies that $\varphi$ is invertible in $C(\mathbf{T})$.

Why is invertibility in $B(H^2) + K(H^2)$ imply invertibility in $\mathbf{A} + K(H^2)$.

---

**Lemma.** Suppose that $\varphi$ is an invertible function in $C(\mathbb{T})$. Then, there is a unique $n \in \mathbb{Z}$ so that $\varphi = \varepsilon\_n  e^\psi$ for some $\psi \in C(\mathbb{T})$.

**Proof.** Suppose that $\\\| 1- \varphi \\\| < 1$. Then, we can take $\ln \circ \varphi$, where $\ln$ is the principal branch of the logarithm, as the criterion above ensures that this is well defined. Similarly, if $\varphi, \varphi'$ are both invertible in $C(\mathbb{T})$, such that $\\\| \varphi - \varphi' \\\| < \\\| \varphi^{-1} \\\|^{-1}$, then $\varphi' = \varphi e^{\psi}$ for some $\psi \in C(\mathbb{T})$.  
Suppose that $\varphi \in C(\mathbb{T})$ is invertible. Since $\Gamma$ is dense in $C(\mathbb{T})$, by the previous paragraph we may assume that $\varphi \in \Gamma$. Then, upon writing $\varphi = \sum_{n \leq N} \varepsilon_n \lambda_n$, we see that $\varepsilon\_N \varphi \in \Gamma^+$, and therefore is a polynomial in $z = \varepsilon\_1$, and therefore we can write $\varphi$ as a product of terms of the form $z - \lambda$, where $\lambda \notin \mathbb{T}$ (since $\psi$ invertible). Thus, it is enough to consider functions of the form $\varphi = z - \lambda$.  
If $|\lambda| < 1$, then $\\\| \varphi - z \\\| = | \lambda | $, and therefore $\varphi = z e^{\psi}$ for some $\psi \in C(\mathbb{T})$. If $|\lambda | > 1$, we have that $\\\| (1- \lambda^{-1} z) -1 \\\| < 1$, and therefore $1 - \lambda^{-1} z$ is of the form $e^{\psi}$ for some $\psi \in C(\mathbb{T})$. From this it follows that any invertible $\varphi \in C(\mathbf{T})$ can be written in the desired form.  
To prove uniqueness it is enough to prove that if $\varepsilon\_n$ is of the form $e^\psi$, then $n=0$. Suppose then, that $\varepsilon\_n = e^\psi$, and consider the map
<center> $$[0,1] \to \mathbb{Z}, \quad t \mapsto \text{ind}(T_{e^{t \psi}})$$ </center>
This function is continuous, with a discrete range and connected domain, and therefore is constant. From this we conclude that $0 = \text{ind}(1) = \alpha(0) = \alpha(1) = \text{ind}(T_{e^\psi}) = \text{ind}(\varepsilon_n) = -n$, hence $n = 0$, as required.

**Remark.** The unique number from the previous lemma is called the *winding number* of $\varphi$ with respect to the origin, denoted $\text{wn}(\varphi)$.

**Theorem.** Let $\varphi$ be an invertible element in $C(\mathbb{T})$. Then, $\text{ind}(T\_\varphi) = - \text{wn}(\varphi)$. Moreover, $T\_\varphi$ is invertible *iff* it is of Fredholm index $0$, *iff* $\varphi = e^\psi$ for some $\psi \in C(\mathbb{T})$.