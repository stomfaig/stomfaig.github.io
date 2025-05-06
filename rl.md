
*A few preliminary remarks.* This page is intended to be a work, where I collect resources, derivations, ideas etc. that I found useful when diving into the world of RL. In particular, it is not a hand-held, explain everything type introduction, but rather one that a person with some mathematical affinity (not much) could understand.
As such, not everything is discussed in detail.

# Short intro to RL

*This section is inspired by [1][1].*

**Action space:** The space of actions a model can take in a given situation.

We can define **policies** given action spaces. A **policy** is simply a mapping (deterministic or not) from the current state of the system to a possible action. If a policy is deterministic, we have:
$$a_t = \mu(s_t)$$
And if it is stochastic:
$$ a_t \sim \mu(\cdot | s_t)$$

**Remark.** We can think of deterministic policies as functions mapping from the state to the action. We can think of non-deterministic policies as classifiers.

The **trajectory** of a system is simply the sequence of states and actions in the world

$$\tau = (s_0, a_0, s_1, \ldots)$$

The way the next state of the system is determined based on the current one is governed by the **natural laws of the environment**:
$$s_{t+1} = f(s_t, a_t)$$
Obviously $f$ can also be stochastic.

### Reward and Return


There are several way in which the reward for a given action is determined:

$$r_t = R(s_t, a_t, s_{t+1})$$
$$r_t = R(s_t, a_t)$$
$$r_t = R(s_t)$$
The main difference here is that what actually determines the reward. In the first case, the current state, the action and the next state all matters. Maybe the next state doesn't matter (2nd case). Or maybe the only thing that matters is the current state. One can easily come up with "real world" examples to all these.  
There are also different ways in which we can define the return of trajectory:
$$R(\tau) = \sum_{t}^T r_t$$
$$R(\tau) = \sum_{t} \gamma^t r_t$$
these are for the cases when the time horizon is finite, or infinite respectively.

### The Goal

We are now in position to state the goal of RL. Given a model of the universe, consider a policy $\pi$ and the distribution $P(\tau|\pi)$, which determines how likely a certain trajectory is given the policy and the laws of the universe. Then, our aim is to solve the problem

$$\pi^* = \arg \max_\pi E_{\tau \sim P(\cdot | \pi)} [T(\tau)]$$

Where the solution $\pi^*$ is called the optimal policy.

### A few useful quantities

Here we list the definition of a few quantities, that are useful when comparing policies.

$$\textbf{On-policy value function:} \quad V_\pi(s) = E_{\tau \sim \pi}[R(\tau) | s_0 = s]$$
$$\textbf{On-policy action-value function:} \quad Q_\pi(s, a) = E_{\tau \sim \pi}[R(\tau) | s_0 = s, a_0 = a]$$
$$\textbf{Optimal value function:} \quad V_*(s) = \sup_{\pi} E_{\tau \sim \pi}[R(\tau) | s_0 = s]$$
$$\textbf{Optimal action-value function:} \quad Q_*(s, a) = \sup_{\pi} E_{\tau \sim \pi}[R(\tau) | s_0 = s, a_0 = a]$$

One can "peel off" the first action in the "value type"- and the first state change of the world in the "action-value" type equations, to get a self-consistency condition between these quantities. These are called the **Bellman equations**.  

We can also define the **advantage function** of an action with respect to a policy $\pi$:
$$A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)$$

# Policy optimization methods.

The methods we are going to encounter are all based on iteratively improving a policy that we currently have to a "better one". To make the subject of the discussion more clear, we more explicitly clarify the objectives of the iterative processes we are to constrcuct: 

1. We our method to increase some performance measure of the policy every step.  
2. We want to be able to verify that our method is improving the quantity that it is trying to increase.  
3. We want to be able to investigate the asymptotic behavior of the method.  


## Some warmup.

### Exact policy iteration.

The basic idea is the following. If we have access to the exact value function, given a policy $\pi$, we can compute $Q_\pi(s, a)$ explicitly for each $(s,a)$ pair, and create a new deterministic policy $\pi'$, such that
$$\pi'(a;s) = \text{$1$ iff $a = \arg \max_a Q_\pi(s,a)$, $0$ otherwise}$$

### Policy gradient.

To take the theory out for a spin, consider the following. Let $\pi_\theta$ be a parametric family of policies. Suppose that the initial state of the universe is determined by some initial distribution $\rho_0$. Then, our aim is to find the policy which would maximize the expression
$$J(\theta) = E_{\tau \sim \pi_\theta} [R(\tau)]$$
An intuitive way to do this would be to compute the gradient $\nabla_\theta J(\theta)$, and perform some form of gradient optimization method.  
Let us start by investigating how changing theta affects the evolution of the universe by thinking about $\nabla_\theta P(\tau | \theta)$. By the _"log-derivative trick"_:
$$\nabla_\theta P(\tau | \pi_\theta) = P(\tau | \pi_\theta) \left( \nabla_\theta \log P(\tau | \pi_{\theta}) \right) $$
The probability of a trajectory $\tau = (s_0, a_0, \ldots)$ given an initial distribution $\rho_0$ is given by:
$$P(\tau | \theta) = \rho(s_0) \prod_i P(s_{i+1} ; s_i, a_i) \pi_{\theta}(a_i | s_i)$$
This way
$$\log P(\tau | \theta) = \log \rho (s_0) + \sum_i \log P(s_{i+1}; s_i, a_i) + \log \pi_{\theta}(a_i|s_i)$$
So if now we take the derivative of this with respect to $\theta$, all the $P(s_{i+1}; s_i, a_i)$ terms will vanish, since they are not dependent on the parameter $\theta$. Hence:
$$\nabla_\theta P(\tau | \theta) = P(\tau | \theta) \left( \sum_i \nabla_\theta \log \pi_{\theta}(a_i | s_i) \right)$$
So then, 
$$\begin{align*} 
    \nabla_\theta J(\theta) &= E_{\tau \sim \pi_\theta} [R(\tau)]\\
    &= \nabla_\theta \int_\tau P(\tau | \theta) R(\tau)\\
    &= \int_\tau P(\tau | \theta) \left( \sum_i \nabla_\theta \log \pi_{\theta}(a_i | s_i) \right) R(\tau)\\
    &= E_{\tau \sim \pi_\theta} [\left( \sum_i \nabla_\theta \log \pi_{\theta}(a_i | s_i) \right) R(\tau)]
\end{align*}$$

## Approximate value functions methods.

One major drawback of Exact Policy Iteration is the fact that it is only guaranteed to improve the performance of the policy, if the new policy has non-negative advantage in every state. This is very hard to guarantee in practice. Thus, instead of trying to optimize $\eta(\pi)$ directly, we are going to define a _surrogate_ objective function. To do this let us fix some notation first.  

Suppose that we have a fixed initial distribution, say $\mu$. We will implicitly assume that this exists, but won't mention that it exists very often. Then, the _discounted visitation frequencies_ are given by:
$$\rho_\pi(s) = P(s_0 = s) + \gamma P(s_1 = s) + \cdots$$
And the _policy advantage_ $\mathbb{A}_{\pi}(\pi')$ of a policy $\pi'$ over another policy $\pi$ as
$$\mathbb{A}_{\pi}(\pi') = E_{s \sim d_{\pi, \mu}} [E_{a \sim \pi'(a;s)} [A_\pi(s, a)]]$$
We claim that the function $L_\pi(\pi_\theta) = J(\pi) + \mathbb{A}_{\pi} (\pi_\theta)$ (as a function of $\theta$) matches $J(\pi_\theta)$ up to first order. One way to prove this, is to consider the expansion:
$$\begin{align*}
    J(\pi_\theta) &= J(\pi) + \sum_t \sum_s P(s_t = s | \pi_\theta) \sum_a \pi_\theta(a | s) \gamma^t A_\pi(s, a)\\
    &= \cdots\\
    &= J(\pi) + \sum_s \rho_{\pi_\theta}(s) \sum_a \pi_\theta(a|s) A_\pi(s,a)
\end{align*}$$
And then take the derivative and compare it to the derivative of $L$.

**Exercise.** Fill the gaps in the previous expansion.  

**Exercise.** Using the outline above, or in any other way, prove that $L_\pi(\pi_\theta)$ agrees with $J(\pi_\theta)$ up to first order.

There is a natural question to ask now:

> We could simply optimize $J$ by substituting it with $L$, and using some sort of gradient based method. But how long should our step size be? Is there any way to use the fact that $L$ and $J$ are not just arbitrary functions, but have a fair amount of structure due to the fact that they arise as RL reward functions?

### Mixture policies.

*The idea of this section was originally published in [here][2]*.

Let's try to apply the previous idea in the simples setting, i.e. along a line.
$$\pi_{\text{new}}^{\alpha}(a;s) = (1 - \alpha) \pi(a;s) + \alpha \pi'(a; s)$$
Where $0 \leq \alpha \leq 1$. We have:

$$\begin{align*}
J(\pi_\text{new}^\alpha) &= L_\pi(\pi') + O(\alpha^2)\\
&= J(\pi) + \mathbb{A}_{\pi}((1- \alpha) \pi + \alpha \pi') + O(\alpha^2)\\
&= J(\pi) + \alpha \mathbb{A}_{\pi}(\pi') + O(\alpha^2)
\end{align*}$$
That is:
$$\frac{\partial J}{\partial \alpha} \bigg|_{\alpha = 0} = \frac{1}{1-\gamma} \mathbb{A}_{\pi, \mu}(\pi')$$
Thus a positive advantage implies the existence of a sufficiently small $\alpha$ such that the policy $\pi_{\text{new}}$ is better than $\pi$.


**Theorem.** Let $\mathbb{A}$ be the policy advantage of $\pi'$ with respect to $\pi$ and the starting distribution $\mu$, and let
$$\varepsilon = \max_s |E_{a \sim \pi'(a, a)}[A_\pi (s, a)]|$$
Then, for every $\alpha \in [0,1]$:
$$\eta_\mu(\pi_{new}) - \eta_{\mu}(\pi) \geq \frac{\alpha}{1 - \gamma} \left( \mathbb{A} - \frac{2 \alpha \gamma \varepsilon}{1- \gamma(1 - \alpha)} \right)$$
 
### Beyond mixture policies - TRPO

*The idea of this section was originally published in [here][3]*.

Fundamentally, we are motivated by the following thought, that extends the result of the previous section:

> Surely, if there is a new candidate policy $\pi'$, such that it has positive advantage over $\pi$, and is sufficiently close (in some sense), then $\pi'$ must be a better policy than $\pi$.

We are going to introduce two distances. One of them is useful for proving a bound similar to ..., and the other is useful for practical computations.

**Definition.** The **total variation divergence** between two discrete probability distributions $p, q$ over the same set is given by:
$$D_{TV}(p || q) = \frac{1}{2} \sum_i |p_i - q_i|$$
And between policies:
$$D_{TV}^\text{max}(\pi, \pi') = \max_s D_{TV} \big[ \pi(\cdot | s), \pi'(\cdot | s) \big]$$

We then have the following theorem, very much in the vein of the corresponding theorem for Mixture policies:

**Theorem.** Let $\alpha = D_{TV}^{\text{max}}(\pi_{\text{old}}, \pi_{\text{new}})$. Then:
$$\eta(\pi_{\text{new}}) \geq L_{\pi_{\text{old}}}(\pi_{\text{new}}) - \frac{4 \varepsilon \gamma}{(1 - \gamma)^2} \alpha^2$$

Now, in practice computing the total variation divergence is not always feasible. It turns out though that it relates to the KL-divergence in a very useful way:
$$D_{\text{TV}}(p || q)^2 \leq D_{\text{KL}}(p || q)$$
And we set
$$D_{\text{KL}}^{\text{max}} (\pi, \tilde{\pi}) = \max_s D_{\text{KL}}(\pi(\cdot | s) || \tilde{\pi}(\cdot | s))$$
Which yields the following form of the above theorem:
$$\eta(\tilde{\pi}) \geq L_{\pi}(\tilde{\pi}) - \frac{4 \varepsilon \gamma}{(1 - \gamma)^2} D_{\text{KL}}^{\text{max}}( \pi, \tilde{\pi})$$

An interesting remark to note here.

> We could consider the policy optimization iteration given by the pseudocode:  
> ```
> while (not converged):  
>   pi_new = argmax_pi L_{pi_i}(pi) - CD_KL_max (pi_old, pi)
> ```
> Which can easily be shown to monotonically improve the policy in every iteration. The interesting thing is to note the similarity between this method and [proximal methods][4] in optimization theory. It turns out, that if one simply optimizes this objective function with the constant $C$ as suggested above, the step sizes of the method end up being really small. Instead, one can consider a [trust-region type method][5]:
> ```
> while (not converged):  
>   pi_new = argmax_pi L_{pi_i}(pi)
>       such that CD_KL_max (pi_old, pi) <= delta
> ```

One final thing to note is that one rarely encounters TRPO in the form we have just described. The following form of the objective is used:

### Almost TRPO: PPO

## Implementations and their consequences.

### Generalized advantage estimation.

### GRPO


