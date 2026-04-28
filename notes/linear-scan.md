---
title: Linear scan and the benefits of SSA
date: 2026-03-19
description: SSA form is a standard intermediate representation used in most modern compilers. Beyond its practical convenience, it has a remarkable theoretical property — it makes the register allocation problem, NP-hard in general, solvable optimally in polynomial time.
---

SSA (static single assignment) is a well-known concept in IR design. It is defined by the characteristic, that each value is assigned exactly once. For example:

```
x = y
use(x)
x = 2 * y
use(x)
x = 3 * y
```
Would become
```
x_1 = y
use(x_1)
x_2 = 2 * y
use(x_2)
x_3 = 3 * y
```
This seemingly artificial modification has deep implications for compiler design, which are not immediately obvious. In this article, I will outline some theoretical motivation why SSA form is so useful.

## Register allocation

One seldom thinks about machine registers when writing any high level code. For compilers this is a much more prominent issue, since the original code, which might be using hundreds of values at a given point will have to be lowered to a machine, that might only have 10 or 20 registers. 

The process of mapping semantic values to actual registers is called **register allocation**. The challenge of register allocation lies in that there are many competing needs to be tackled.  

1. Values stored in registers can be accessed much more quickly than ones in the memory. Thus values that are often used should ideally be stored in a register for as long as possible.  
2. Since the number of values in the program is large, at any given point only a subset of them should be stored in a register.

Note also that this problem has to be answered for each line of code. That is, the challenge is not just that there are competing needs, but also the fact that the solution to the problem changes between different program points.

### Liveness and the Interference graph

We start by formalizing the ideas presented in the previous section.

::: definition
Given a CFG, we define a value `v` stored in a variable `V` to be **live** at a program point `p`, if There is a path in the CFG from `p` to a use of `V` along which the variable `V` is not assigned a new value.
:::

The notion of liveness is important, because it captures exactly the notion of _this value is going to be used later, so keep it in a register until the use if possible_. It follows, that at any given point, the pool of candidates for values to be stored in registers is exactly the set of live values at that point. This motivates the following definition.

::: definition
Two values in a program are said to **interfere**, if there is a program point, at which they are both live.  
The **interference graph** of a program is the graph $IFG = (V, E)$, with vertex set $V$ is the set of values in a program, with edges between two nodes if they interfere.
:::


### How are registers allocated?

Using the terminology developed in the previous section, we can formulate one way for allocating registers in for a program. Consider some mapping $f: V \to [N]$, where $[N]$ is just some set with size $N$, for example, the set of registers of a machine with $N$ registers. One can think of this as coloring the vertices of the graph with $N$ colors. We say an assignment $f$ is valid, if $f(v) \neq f(u)$ whenever $(u, v) \in E$. This exactly corresponds to the idea, that no two values which are live at the same program point should be mapped to the same register.

This allows us to sketch a simple register allocation algorithm:
```
1. Compute the IFG
2. Color the nodes of the IFG
3. When a value $v \in V$ is computed, store it in the register given by $f(v)$
```

::: warning
Before we proceed, it is important to note that there are many aspects of register allocation that are omitted here:

- In some cases not all registers are uniform, and there might be restrictions on what register an instruction can access.
- In real code, we often can't color the inference graph, and other methods like _live interval splitting_, or _spilling_ are required. 
None of these are super complicated topics in themselves, but the point is that they all have to be accounted for, but for the sake of simplicity, we do not discuss them in detail.
:::

In this most general case, it is still not clear how one might obtain a coloring of the graph. One simple approach is greedy coloring. We process nodes in some fixed order, and assign each node the lowest-numbered color not already used by any of its neighbors that have been colored so far.

```
Given an ordering v_1, v_2, ..., v_k of the nodes:
For each v_i:
  Assign the smallest color c not used by any neighbor v_j (j < i) of v_i.
```

This always produces a valid coloring, though the number of colors used depends on the ordering chosen.

## Live ranges and the nesting number

_To my best understanding, the definition and usage of the nesting number (which is presented in this section) is not usual in the literature, so if you plan to reuse it, please refer to this article._

So far we only talked about register allocation, and with good reason. SSA form will play a crucial role in turning the above algorithm into something that is efficient in practice. The crucial idea will be to discover that certain nodes cannot possible be connected in the interference graph, which will give it a special structure. We will need some more groundwork for this, however.

::: definition
Given a basic block $B$ in the CFG, the **nesting number** $\eta(B)$ is the number of ancestors of $B$ in the dominator tree with out-degree greater than one, and are not post-dominated by $B$. Equivalently, $\eta(B)$ counts the branch points that dominate $B$, but are not re-merged before $B$.
:::

The entry block has $\eta = 0$. A block reachable only through one branch of a conditional has $\eta = 1$, and so on. The key theorem is:

::: {.theorem data-title="1"}
In a program in SSA form, let $v$ be a variable defined in block $B_{\mathrm{def}}$. Then $v$ is live only in blocks dominated by $B_{\mathrm{def}}$. In particular, every block in $v$'s live range has nesting number $\eta \geq \eta(B_{\mathrm{def}})$.
:::

*Note.* We do not provide any proofs in this article as they are quite self-explanatory (except for Theorem 5). The reader is encouraged to work out these proofs.

The idea of this proof, is that if a variable is defined in one branch of a conditional, then it cannot be alive once the branches of the conditional re-merge, since the execution could have taken the other branch, leaving the value undefined. The hard thing is that in general branching structures can be quite complicated. The concept of branching numbers makes analysing branches simpler, courtesy of Theorem 3 below.

We first need a structural lemma about how branching interacts with the dominator tree. Given $B_1$ and $B_2$ incomparable in the dominator order, define their least common descendant $\mathrm{LCD}(B_1, B_2)$ as any block forward-reachable from both $B_1$ and $B_2$ whose proper dominators are not forward-reachable from both (i.e., the first join point of the two paths).

::: {.theorem data-title="2"}
Given two program basic blocks $B_1$ and $B_2$ such that $B_1 \cancel{\leq} B_2$ and $B_2 \cancel{\leq} B_1$, if $p = \mathrm{LCD}(B_1, B_2)$ exists then

$$
\eta(p) < \min(\eta(B_1),\, \eta(B_2))
$$

Moreover, neither $B_1$ nor $B_2$ dominates $p$.
:::

Combining Theorems 1 and 2 we get:

::: {.theorem data-title="3"}
Given definitions in basic blocks as $d_1: B_1$ and $d_2 : B_2$, such that $B_1 \cancel{\leq} B_2$ and $B_2 \cancel{\leq} B_1$, then $d_1$ and $d_2$ **cannot** interfere.
:::

This is exactly the promised special structure: when checking which other variables a given variable can interfere, it is sufficient to check its ancestors and descendants. 

## Linear scan

Using the property we deduced in the previous section, if we traverse the CFG in a depth-first order, we can exploit the non-interference patterns. More accurately, given a CFG, perform a depth-first traversal, and construct the following code listing:

```
line 1: Code from the BB traversed first
...
line N: Code from the BB traversed second
...
line M: ...
```

This way, for each value, rather than specifying program points in `BB:line` fashion, we can simply specify the live region of each value by the interval `[line of definition, line of last use)`. This then also allows us to construct another sort of Interference graph, which we will call the _linearized interference graph_ (or LIG for short). In this graph two values are connected iff their live regions intersect. The key result is the following:

::: {.theorem data-title="4"}
The linearized interference graph associated with the code obtained from any depth-first ordering is exactly the interference graph of the code as defined previously.
:::

Note also, that by construction the LIG (and therefore the interference graph) has the structure of an [interval graph](https://en.wikipedia.org/wiki/Interval_graph), which is also a sub-family of the well known class of [chordal graphs](https://en.wikipedia.org/wiki/Chordal_graph), which are in turn [perfect graphs](https://en.wikipedia.org/wiki/Perfect_graph). While these don't have any immediate relevance, the following theorem (that applies for each perfect graph) does:

::: {.theorem data-title="5"}
Any perfect graph can be coloured with as many colours as the size of the maximal fully connected component (clique).
:::

The proof of this theorem is not too interesting for our purposes, as it lives in vastly more general realm than our compiler specific work here, so we will not bother with thinking about it. A more specific result, that the reader should be able to easily prove, is to show that one needs at least as many colours as the number of nodes in the smallest clique. One can then show that the algorithm below uses exactly as many colours as the largest clique, which implies that it is optimal.

Compared to perfect graphs, which only guarantee the existence of a colouring, the even more special form of our interval graph allows us to directly construct a colouring with optimally many colours in polynomial time. This is the famous linear scan algorithm.

```
1. Sort the values by their liveness intervals, according to the following order:
  - [a, b) ≤ [c, d) whenever a < c
  - [a, b) ≤ [a, c) whenever c ≤ b
2. Maintain a set `active` of intervals that have started but not yet ended,
   and a set `free` of available register names.
3. For each interval [a, b):
     - Remove from `active` all intervals [x, y) with y ≤ a;
       return their registers to `free`.
     - Assign to [a, b) the smallest register in `free` (allocate a fresh one
       if `free` is empty), and add [a, b) to `active`.
```

As you might have noticed, this algorithm can be implemented in polynomial time, which means that we can colour the interference graph of code in SSA form in polynomial time -- a huge improvement from the NP-hard complexity and the guarantee-less heuristics that we saw for completely general classes.  

Interestingly, the algorithm is really simple. However, note that the reason why it works is rooted the extensive theory that we developed.

## Conclusion

We started with a problem that is NP-hard in full generality: given a program and a machine with $N$ registers, assign values to registers so that no two simultaneously live values share one. The standard formulation reduces this to graph coloring, and graph coloring on arbitrary graphs is NP-complete.

SSA form breaks the hardness. The key structural insight is that SSA live ranges are confined to the dominator subtree of their definition point (Theorem 1), and two values defined in incomparable blocks can never interfere (Theorem 3). This means that when the CFG is flattened by a depth-first traversal, every live range becomes a contiguous interval — and the interference graph becomes an interval graph. Interval graphs are perfect, so their chromatic number (the number of colours needed to colour them) equals the number of nodes in the largest clique in the graph, and the linear scan algorithm colors them optimally in $O(n \log n)$ time.

In other words: the single-assignment discipline imposes enough structure on the program that a problem which is intractable in general becomes efficiently and optimally solvable.

---

*If you spot an error or have a question, feel free to open an issue or send me an email.*