---
layout: post
title: "A brief summary of Compact and finite rank operators"
mathjax: true
categories: misc
---

*Disclaimer: My notes heavily influenced by 'An introduction to the theory of C\*-algebras' by G. J. Murphy, and many proofs and developments are the exact ones in the book. These documents are intended to help for those reading the book, by providing more detail in some cases, or by organizing the material in a different way.*

*This article is not complete yet*

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

**Definition.** invariance and irreducibility
 
**Theorem.** Let $A$ be a C\*-algebra acting irreducibly on a Hilbert space $H$, and having non-zero intersection with $K(H)$. Then, $K(H) \subset A$.

**Proof.**
