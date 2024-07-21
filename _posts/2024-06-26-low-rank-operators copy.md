---
layout: post
title: "Some interesting results about finite rank- and compact operators"
mathjax: true
categories: misc
---

*Disclaimer: My notes heavily influenced by 'An introduction to the theory of C\*-algebras' by G. J. Murphy, and many proofs and developments are the exact ones in the book. These documents are intended to help for those reading the book, by providing more detail in some cases, or by organizing the material in a different way.*

---

Given a Hilbert space $H$, let $F(H)$ denote the finite-rank operators on $H$, and $K(H)$ the subspace of compact operators. We have the following.

**Theorem.** $F(H)$ is dense in $K(H)$.

**Proof.** Clearly $F(H)$ and $K(H)$ are both self-adjoint, hence given $a \in K(H)$ the unique self-adjoint elements $c, b$ such that $a = b + ic$ are also in $K(H)$. Thus, it is sufficient to prove the statement for self-adjoint elements of $K(H)$.  
Given $u \in K(H)$ self-adjoint, $u$ is diagonalisable and its eigenvalues tend to 0. From this, one can construct diagonal approximations to $u$ of finite rank that tend to $u$ in norm.

We start by introducing the following very convenient notation. Let $H$ be a Hilbert space, and let $x, y \in H$. Then, define $x \otimes y$ as follows:

<center> $$ (x \otimes y)(z) = \langle z, y \rangle x $$ </center>

**Theorem.** If $H$ is a Hilbert space, then $F(H)$ is linearly spanned by the rank-one projections.

**Proof.** Since $F(H)$ self-adjoint, it is sufficient to prove the statement for $u \in F(H)$ self-adjoint. Further, by polar decomposition $|u| \in F(H)$, so without loss of generality the positive and negative parts of $u$ are in $F(H)$. Hence, we may assume that $u \geq 0$.  
Note that $u(H)$ is a finite dimensional subspace of $H$, thus a Hilbert space itself, with orthonormal basis $e_1, \ldots, e_n$, say. Consider the projection $p = \sum_{j=1}^{n} e_j \otimes e_j$ of $H$ onto $u(H)$. Then, $u = pu$, and therefore $u^{1/2} = (pu)^{1/2}$. Both sides are self-adjoint, so
<center> $$ u = (u^{1/2})^2 = (u^{1/2})^* u^{1/2} = $$  there is a mistake here I think</center>
Once the above line if figured out, the proof should go fine...

**Theorem.** If $I$ is a non-zero ideal in $B(H)$, then $I$ contains $F(H)$.

**Proof.** Let $I$ be a non-zero operator in $I$. Then $u(x) \neq 0$ for some $x \in I$. Consider a rank-one projection $p = y \otimes y$ for some $y \in H$ of unit length. We can find $v \in B(H)$ such that $vu(x) = y$. Then,
<center> $$ vu \big( x \otimes x\big) (vu)^* = y \otimes y $$ </center>
Therefore all rank-one projections are in $I$, as $u \in I$. Now apply the previous theorem.

**Remark.** In the above proof, it might seem a bit goofy to construct $y \otimes y$ from $x \otimes x$, and a lot of other things. To me it helps to understand and memorise this proof by thinking about the fact, that the *only* thing available to us to construct $y \otimes y$ is the fact that $u(x) \neq 0$, and that in the construction of $y \otimes y$, $u$ must appear somewhere.

Recall that given $K$ a vectorsubspace of a Hilbert space $H$, we say that $K$ is invariant for $a \in B(H)$, if $a(H) \subset H$.

**Definition.** We say that a closed vectorspace $K$ of a Hilbert space $H$ is *invariant* for a subset $A \subset B(H)$, if it is invariant for every operator in $A$. If $A$ is a C\*-subalgebra of $B(H)$, we say that $A$ acts irreducibly on $H$, if the only closed irreducible subspaces for $A$ are $0$ and $H$.
 
For the following theorem we need a lemma about general properties of compact operators. For the proof I recommend reading any introductory functional analysis textbook.

**Lemma.** Suppose that $u$ is compact operator on a Banach space. Then,
1. if $\lambda \in \sigma(u)$, then $\lambda$ is an eigenvalue of $u$,
2. any $0 \neq \lambda \in \sigma(u)$ is an isolated point of $\sigma(u)$,
3. for any $a \neq \lambda \in \sigma(u)$, the eigenspace $\ker (u- \lambda)$ is finite dimensional.

We also use the following result:

**Lemma.** If $A$ is finite dimensional C\*-algebra, then it is spanned by its projections.

**Proof.** Suppose that $A$ is Abelian. Then, without loss of generality $A \equiv C_0(Omega)$ for some $\Omega$ compact Hausdorff by the Gelfand representation.  
Note that, for $\omega_1, \ldots, \omega_n \in \Omega$ distinct, the characters $\tau\_{\omega_i}$ given by
<center> $$\tau_{\omega_i}(f) = f(\omega_i) \quad f \in C_0(\Omega)$$ </center>
are linearly independent, so $\Omega$ is finite and discrete. This means that the indicators of singleton sets (which are projections) span $C_0(\Omega)$, and thus the statement is proved.  
In the non-abelian case, we know that $A$ is linearly spanned by its self-adjoint elements. However, applying the previous arguments to $C(a)$ for any self-adjoint $a \in A$, we see that $a$ is a linear combination of projections in $A$; and thus the statement follows. $\blacksquare$

**Theorem.** Let $A$ be a C\*-algebra acting irreducibly on a Hilbert space $H$, and having non-zero intersection with $K(H)$. Then, $K(H) \subset A$.

**Proof.** We start by proving that there are finite rank projections in $A$.  
The intersection $A \cap K(H)$ is a self-adjoint non-empty set, so it contains a self-adjoint non-zero element, say $u$. Since $u$ self-adjoint, $r(u) = \| u\| \neq 0$, thus there is $0 \neq \lambda \in \sigma(u)$. This implies that $\lambda$ is an eigenvalue of $u$ (since $u$ compact), and $\lambda$ is an isolated point of the spectrum of $u$, so the function $f = \mathbb1\_{\\\{\lambda\\\}}$ is continuous, and $p = f(u)$ is a projection in $A$. Since $f \neq 0$, $p \neq 0$.  
If $z: \sigma(u) \hookrightarrow \mathbb{C}$ is the inclusion map, then $(z - \lambda)f = 0$, so $(u - \lambda)p = 0$, so $p(H) \subset \ker (u - \lambda)$. We since $u$ compact, $\ker (u- \lambda)$ is finite dimensional, so $p(H)$ is a finite rank projection.  

Let $q$ be a projection in $A$ of minimal finite rank. Write $q = \sum_{j=1}^n e_j \otimes e_j$, where $e_j \in H$ are unit length vectors. Then, for any $u \in A$,
<center> $$pup = \sum_{j,k = 1}^{n} (e_j \otimes e_j) u (e_k \otimes e_k) = \sum_{j,k = 1}^{n} \langle u(e_k), e_j \rangle \ e_j \otimes e_k$$ </center>
hence $qAq$ is finite dimensional, therefore it is the linear span of its projections, by a preceding lemma. Since $q$ is assumed to have minimal rank, the only projections in $qAq$ must be $0$ and $q$, and $qAq = \mathbb{C} q$.  
Consider $y \in q(H)$, and set $K = \text{cl} \ \\\{ u(y) : u \in A \\\}$. Then $K$ is a vector subspace of $H$, that is invariant for $A$ [Since if $u, v \in A$, then $v(u(y)) = uv(y)$, and $uv \in A$], which is clearly non-empty, since $y = q(y) \in K$. Since $A$ is irreducible, $K = H$. For every $x \in q(H)$, there is a sequence $(u_n) \in A$ such that $x = \lim_n u_n(y)$, and since $x,y \in q(H)$, $x = q(x), y = q(y)$, so $x = \lim_n q u_n q (y)$. Since $qAq = \mathbb{C}q$, there is $\lambda_n$ so that $qa_n q = \lambda q$, but then $x \in \mathbb{C}y$, and hence $q(H) = \mathbb{C}y$, and since $q$ is a projection, $q = y \otimes y$.  
Now take an arbitrary unit vector $x \in H$. As above, there is a sequence $(u_n)_n$ such that $\lim_n u_n(y) = x$, and hence
<center> $$ x \otimes x = \lim_n u_n(y) \otimes u_n(y) = \lim_n u_n (y \otimes y ) u^*n = \lim_n u_n q u^*_n $$ </center>
hence $x \otimes x \in A$, and so all rank-one projections are in $A$. This implies that $K(H) \subseteq A$ by a previous theorem. $\blacksquare$

---

*Last revision: 21st of July 2024*
