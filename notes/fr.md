---
title: Why from-read is less weird than it seems
date: 2026-04-28
description: Weak memory models often include a from-read term in their definition of the communication relation, which unlike the other terms in that same definition, is much less clear cut. In this note we investigate what meaning should be attached to fr.
---
$$\newcommand{\com}{\mathtt{com}}\newcommand{\rf}{\mathtt{rf}}\newcommand{\co}{\mathtt{co}}\newcommand{\fr}{\mathtt{fr}}\newcommand{\pr}{\mathtt{pr}}$$

I have recently read [Jade Alglave's A formal hierarchy of weak memory models](http://www0.cs.ucl.ac.uk/staff/j.alglave/papers/fmsd12-41-2.pdf)[^disc], and, while considering the definition
$$\com = \mathtt{rf} \cup \mathtt{co} \cup \mathtt{fr}$$
I finally acted upon my urge to actually deeply understand why this relation is a good way to capture the "communication" of different memory actions. 

A concurrent execution is modelled by a set of *events* $E$ — each a read or a write to some memory address — with $[R] \subseteq E$ denoting the reads and $[W] \subseteq E$ the writes. An *execution witness* is a pair $(\rf, \co)$ of relations on $E$ recording how those events interact.

Let's recall what each component of $\com$ actually means.

- $\rf$: The reads-from relation. $(w, r) \in \rf$, if the value returned by $r$ was originally set by the write event $w$
- $\co$: The coherence relation, defined as $\co = \bigcup_x \co_x$ where each $\co_x$ is a total order on writes to address $x$. Formally, $(w_1, w_2) \in \co_x$ if both $w_1, w_2$ write to $x$ and $w_1$ precedes $w_2$.
- $\fr$: The from-reads relation. $(r, w) \in \fr$, if there is a write $w'$ such that $(w', r) \in \rf$ and $(w', w) \in \co$.

The definitions of $\rf$ and $\co$ are transparent enough, they clearly outline some sort of temporal constraint, or maybe even flow of information, but the definition of $\fr$ seems ad-hoc. Especially since it does not capture any real flow of information, or _communication_, as the name of the $\com$ relation might suggest. In particular, one might wonder why do we not define another relation, like:

- $\pr$: The preceding read relation. $(w, r) \in \pr$, if there is a $w'$ such that $(w, w') \in \co$ and $(w', r) \in \rf$.

and also include it in the relation $\com$. After all, it seems to be a "past" analogue of the $\fr$ relation.

### $\pr$ is strictly weaker than $\fr$

The first observation is to notice that $\pr$ in itself is a really bad candidate to replace $\fr$. To see why, recall that $\com$ is usually used as part of an acyclic condition like `acyclic com | po-loc`. An alternative way to express this would be to require the relation `(com | po-loc)*` not to be reflexive for any event. In particular, this means, that any changes to $\com$, that do not also change `com*` will not actually change the behaviour of $\com$ in the settings were we usually use it.

The upshot then comes by noting that `(com | pr)* = com*`, since $\pr = \co ; \rf$. That is, the $\pr$ relation is included in the transitive closure of $\com$ -- in fact even in the transitive closure of $\co \cup \rf$. But one could ask, does $\fr$ buy us anything? Maybe it is also useless.

But $\fr$ is in fact not useless. One easy way to see this, is to consider the "signature" of the $\co$ and $\rf$ relations: any $\co$ relation is of the form $(W, W)$, i.e. is composed of two writes. Formally one would write $\co \subset [W] ; [W]$. Similarly, $\rf \subset [W] ; [R]$. One can then prove that $(\co \cup \rf)* \subset [W] ; [W] \cup [W] ; [R]$, which implies that $\fr$ cannot be contained there, since any $\fr$ relation is an element of $[R] ; [W]$.

### Why is $\fr$ useful

So then, concluding that $\fr$ is indeed independent of $\co$ and $\rf$, we can now ask, what do $\fr$ events mean, or, what is their inclusion win us in terms of understanding data flow. In investigating this question, we are going to fix a memory location $x$ from this point on (this can be done without loss of generality, since both $\co$ and $\fr$ only relate events that act on the same address.)

Let's start by unpacking again what it means for an event to be in $\fr$. Let $(r, w) \in \fr$, and let $w' \in [W]$ such that $(w', r) \in \rf$ and $(w', w) \in \co_x$. A priori, if we'd draw a directed graph of what this means, we see that (1) $w'$ precedes $w$ in $\co_x$ (and therefore in the local ordering at $x$), and (2) that $w'$ precedes $r$ in $\rf$ (and thus also in the local order at $x$). It might seem like we cannot make any further inferences. This would not be surprising assuming that we are in some generic partial order, since all we know is that $w'$ precedes both $r$ and $w$, but no more.

However, the semantics of the domain do imply more ordering information: $r$ must precede $w$. Suppose for contradiction that $r$ does not precede $w$, i.e. $w$ precedes $r$ in the execution. By the *read-write coherence* axiom[^rwc], $r$ would be forced to read from some $w'' \in [W]$ with $(w, w'') \in \co_x$ (i.e. something at least as recent as $w$), not from $w'$. Since $r$ reads from exactly one write, so $w'' = w'$. But then $(w', w) \in \co_x$ and $(w, w') \in \co_x$, contradicting that $\co_x$ is acyclic.

So $\fr$ seems to capture this additional piece of timing information: that a read from $w'$ must precede any other writes coherence-after $w'$.

This is precisely why $\fr$ earns its place in $\com$. The signature argument showed that $\fr$ brings new information on top of $\co$ and $\rf$. This new information turns out to be what the read-write coherence axiom tells us about the order of reads and writes.

So a good way to understand $\com$ might be to think of it as all the temporal relations that are imposed per memory location.

### The read-amortised event space

Notice that this additional ordering information can be taken a bit further: it seems that all reads can be slotted nicely between writes. Further, if
$$\forall w \in [W], |\{r \in [R] : (w, r) \in \rf\}| \in \{0, 1\} \tag{1}$$
then for each location, the events associated with the given location would themselves form a total order. Of course the stipulation (1) is needed, because it might not make sense to choose an order for two reads from different processors reading the value written by the same write, so the totality would be lost there. But we can remedy this as follows.

::: definition 
Given a set of events $E$ and an execution witness $(\rf, \co)$, define the relation $\sim$ on events of $E$ by:
$$r_1 \sim r_2 \quad \text{iff} \quad r_1, r_2 \in [R_E] \text{ and } \exists\, w \in [W_E] \text{ s.t. } (w, r_1) \in \rf \text{ and } (w, r_2) \in \rf$$
Then $\sim$ is an equivalence relation. Write $\mathcal{R} = [R_E]/{\sim}$ for the set of equivalence classes; the pair $([W_E], \mathcal{R})$ is the **read-amortised event space** of the execution witness.
:::

Clearly one has to prove a few things here, but they are rather trivial, so we do not bother. This then allows us to state the following theorem.

::: theorem
For a valid execution (in particular, assuming $\com$ is acyclic), and with the read-amortisation process as defined above,

1. $\com$ is compatible with read-amortisation,
2. For each location $x$, the amortised $\widehat{\com}|_x$ extends uniquely to a total order $\hat{\co}_x$ on $W_x \cup \mathcal{R}_x$.
:::

*Proof sketch.* Claim 1 holds because $\sim$ identifies reads by their $\rf$-source, and each component of $\com$ treats same-source reads identically: $\fr$ depends only on the source write, $\rf$ maps the source write to the whole class, and reads do not appear in $\co_x$. For claim 2, write the writes to $x$ in coherence order $w_1 <_{\co_x} \cdots <_{\co_x} w_n$. A read group $G \in \mathcal{R}_x$ reading from $w_i$ must sit after $w_i$ (by $\rf$) and, for $i < n$, before $w_{i+1}, \ldots, w_n$ (by the $\fr$ argument above), forcing $w_i <_{\hat{\co}_x} G <_{\hat{\co}_x} w_{i+1}$; a group reading from $w_n$ is placed after $w_n$ with no upper constraint. This pins down $\hat{\co}_x$ uniquely: since $\co_x$ is a total order on writes and each read group has a unique $\rf$-source, every group occupies exactly one slot, leaving no freedom in the construction. $\square$

### A different presentation of Execution Witnesses

The closing theorem of the previous section implies that execution witnesses could be presented more canonically. Instead of specifying both $\rf$ and $\co$, one can specify a single total order per memory location. 

::: definition
An **amortised event space** is a pair $(W, \mathcal{R})$ where $W$ is a set of write events and $\mathcal{R}$ is a collection of non-empty, pairwise disjoint sets of read events. Write $W_x$ and $\mathcal{R}_x$ for the restrictions to location $x$.

An **amortised execution witness** for $(W, \mathcal{R})$ is a family $\{\hat{\co}_x\}$, where each $\hat{\co}_x$ is a total order on $W_x \cup \mathcal{R}_x$. An ordinary execution witness $(\rf, \co)$ on an event space $E$ induces an amortised execution witness via the theorem above, by taking $(W, \mathcal{R}) = ([W_E], [R_E]/{\sim})$. Conversely, from $\{\hat{\co}_x\}$ one recovers:

- $\co_x$: the restriction of $\hat{\co}_x$ to $W_x$, and
- $\rf$: each read $r$ in a group $G \in \mathcal{R}_x$ reads from the unique write immediately preceding $G$ in $\hat{\co}_x$.
:::

One can now check that going back and forth between the usual-, and amortised setting leaves everything unchanged.

The benefit of this presentation seem to be that we avoid having to rely on opaque notions and recombine something very similar to the amortised execution witness from several pieces.

On the other hand, this unified notion seems to have its own clarity issues. In the original execution witness presentation it makes good sense to have separate relations for semantically different events. This is especially true in algorithmic applications.

[^disc]: Since we skip most of the introductory material, this is the place to go for the fundamentals. And a good read regardless.

[^rwc]: A read cannot observe a write $w'$ if a write coherence-after $w'$ has already occurred before the read.

---

*If you spot an error or have a question, feel free to open an issue or send me an email.*