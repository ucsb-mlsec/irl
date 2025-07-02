# Inverse Reinforcement Learning for LLM

## Preliminaries
A dataset typically contains $\{(q, r, a)\}$ where $q$ is the question, $r$ is the LLM response (reasoning path + prediction), $y$ is the ground truth answer. 

A state s at time step t is defined as a sequence of tokens: $q + r[:t]$. The action a at time step t is the token $r[t+1]$. The state transition function is deterministic, i.e., $s_{t+1} = s + a$. The initial state is $s_0 = q$.

The goal is to learn the reward function $r(s, a)$ that assigns a score to each pair of state and action.

It is easy to observe that, for each question in the dataset, its initial state is different, which means, the state space is huge, and we only have **one** expert demonstration (which is $r$) for each question. 


## Current Implementation

In traditional inverse reinforcement learning, the states and actions are finite, for example, in the 9x9 grid world, the states are the coordinates of the agent, and the actions are the directions the agent can move. Thus we can easily solve the forward and backward pass of the IRL algorithm.

### Why not implement the backward and forward pass?
However, as I stated above, the state space is huge in our case, and we only have one expert demonstration for each question. Thus, it is not necessary to solve the forward and backward pass of the IRL algorithm as **for almost every state-action pair, we only visit once**, which means the visitation frequency $\mu(s, a)$ is equal to $\frac{1}{n}$, the gradient degenerates to the solution 1 which is discussed below. (This is discussed in CS285 lecture 20 (page 21-23).)

Recall the gradient:

$$
\nabla_\theta \mathcal{L} = \underbrace{\mathbb{E}_{\tau \in \pi^*(\tau)}[\nabla_\theta r_\theta(\tau_i)]}_{\text{estimate with the expert samples}} - \underbrace{\mathbb{E}_{\tau \in p(\tau | O_{1:T}, r_\theta)}[\nabla_\theta r_\theta(\tau)]}_{\text{soft optimal policy under current reward function}}
$$

According to the lecture, the first naive solution is:

### Solution 1. Lazy update of the policy model (done)

Improve $p(\tau | O_{1:T}, r_\theta)$ a little bit (e.g., update the policy model by one or few gradient descent steps) and then, sample trajectories $\{\tau_j\}$ from the updated policy to estimate the gradient.

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta r_\theta(\tau_i) - \frac{1}{M}\sum_{j=1}^M[\nabla_\theta r_\theta(\tau_j)]
$$

Since in our cases, we don't want to update the policy model, the sampling algorithm is: **given a base policy model, a reward model, we first select the top k tokens with the highest probability from the base policy model, then we use the reward model to re-rank these tokens, and select the top 1 token that has the highest reward score as the action. We can repeat this process or keep top n tokens when selecting the actions, to obtain $M$ trajectories for one question.**


### Solution 2. Importance sampling (not finished)

The problem of the above solution is that the estimation is biased. One solution is to use importance sampling. The gradient becomes:

$$
\nabla_\theta \mathcal{L} \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta r_\theta(\tau_i) - \frac{1}{\sum_j w_j}\sum_{j=1}^M w_j \nabla_\theta r_\theta(\tau_j)
$$

where $w_j = \frac{p(\tau_j)\text{exp}(r_\theta(\tau_j))}{\pi_{\tau_j}} = \frac{\text{exp}(\sum_t r_\theta(s_t, a_t))}{\prod_t \pi(a_t| s_t)}$ is the importance weight.


## Questions

### Insufficient expert demonstrations and sample size
For every question in the dataset, we have only one expert demonstration, which means $N=1$ for the above equations. We may need sample multiple responses for each question to better estimate the gradient. Another question is, how many samples are required for soft optimal policy under current reward function? Currently I only sample one trajectory by selecting the token with the highest reward score as the action.

### Importance sampling
Regarding implementing the importance sampling, the reward model $r_\theta(\tau_j)$ is easy to calculate, which is the score of the trajectory. $\pi_{\tau_j}$ is the probability of the trajectory under the current policy model, we can estimate it by softmax the reward scores of the tokens in the trajectory (we can discuss it later).

### Reward model
The current reward model is a LLM + multi-layer feed forward NN. The input is the state and action, then we extract the last hidden state of the LLM as the feature of state s and action a. Then we feed the feature into the feed forward NN to get the reward score.

The problem of this implementation is: we can not efficiently calculate the reward for the entire vocabulary. 

One solution is, given the feature of state s and action a, the feedforward NN outputs a score for each token in the vocabulary.

###

1. expert demonstrations are not enough, sample 10 responses for each question
2. policy sampling: softmax as probability, then select, sample n trajectories
3. reward model update: lora + ffn v.s. ffn only
4. reward model: change vocab only v.s. ffn
5. base reward model: qwen-2.5-math
6. update policy
7. baseline: Qwen/Qwen2.5-Math-PRM-7B, Qwen/Qwen2.5-Math-RM-72B
