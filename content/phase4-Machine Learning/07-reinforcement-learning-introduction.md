# Phase 4 · Lesson 7 — Reinforcement Learning (Introduction)

> Prerequisite: Probability, Optimization (Phase 3), Supervised Learning (Lesson 1) — this lesson closes out Phase 4

---

## 1. Introduction

### What is reinforcement learning (RL)?
A learning paradigm where an **agent** learns to make sequential decisions by interacting with an **environment**, receiving **rewards** (or penalties) as feedback, with the goal of maximizing cumulative reward over time — distinct from supervised learning (no labeled "correct action" is given, only a reward signal) and unsupervised learning (there is an explicit objective — maximize reward — not just "find structure").

### Why does it exist?
Many real problems have no dataset of "correct answers" to learn from, only a way to try actions and observe consequences — game-playing, robotics, resource allocation, and (highly relevant to 2026) fine-tuning LLMs via human feedback (RLHF, Phase 6) all share this structure: an agent must learn a good *policy* (strategy) through trial, error, and delayed reward.

### Historical background
RL's mathematical foundation is the Markov Decision Process (Bellman, 1950s dynamic programming). Modern deep RL exploded with DeepMind's DQN (2013-2015, Atari-playing agent) and AlphaGo (2016) — but RL's most consequential 2020s application, by far, is **RLHF (Reinforcement Learning from Human Feedback)**, the technique that turned raw pretrained language models into the aligned, instruction-following assistants (ChatGPT, Claude, and every major LLM product) that define the current AI era — making this lesson a direct prerequisite for fully understanding Phase 6's LLM fine-tuning content.

### Real-world motivation
Every time you interact with an LLM that follows instructions helpfully and avoids harmful outputs rather than just continuing text statistically, you are experiencing the output of an RL process (RLHF or its descendants) applied on top of a pretrained language model — this lesson provides the conceptual foundation for that entire mechanism.

---

## 2. Theory

### Markov Decision Process (MDP) — the formal framework
An MDP is defined by $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:
- $\mathcal{S}$: states, $\mathcal{A}$: actions
- $P(s'|s,a)$: transition probability (next state given current state/action)
- $R(s,a)$: reward function
- $\gamma \in [0,1)$: discount factor (how much future reward is valued relative to immediate reward)

The **Markov property**: the future depends only on the current state, not the full history — a simplifying (and sometimes limiting) assumption that makes the mathematics tractable.

### Policy, value function, and the goal of RL
- **Policy** $\pi(a|s)$: the agent's strategy — probability of taking action $a$ in state $s$.
- **Value function** $V^\pi(s) = E_\pi\left[\sum_{t=0}^{\infty}\gamma^t R_t \mid s_0=s\right]$: expected cumulative discounted reward starting from state $s$, following policy $\pi$.
- **Action-value function (Q-function)** $Q^\pi(s,a)$: expected cumulative reward starting from $s$, taking action $a$, then following $\pi$ thereafter.
- **Goal**: find $\pi^*$ maximizing $V^\pi(s)$ for all states — the **optimal policy**.

### Exploration vs. exploitation
An agent must balance **exploiting** its current best-known strategy (to accumulate reward) against **exploring** less-tried actions (to discover potentially better strategies) — a fundamental tradeoff with no free lunch, formalized classically via the multi-armed bandit problem (a simplified, stateless special case of RL).

### Model-based vs. model-free RL
- **Model-based**: the agent learns (or is given) the environment's transition dynamics $P(s'|s,a)$ and plans using them.
- **Model-free**: the agent learns a value function or policy directly from experience, without ever explicitly modeling transition dynamics — Q-learning and policy gradient methods (below) are model-free, the dominant paradigm in modern deep RL.

---

## 3. Mathematical Foundations

### The Bellman Equation (the central recursive relationship of RL)
$$
V^\pi(s) = \sum_a \pi(a|s)\sum_{s'} P(s'|s,a)\left[R(s,a) + \gamma V^\pi(s')\right]
$$
This recursive structure — a state's value equals immediate reward plus the discounted value of wherever you end up next — is exactly a **dynamic programming** recurrence (Phase 1 Lesson 4's DP concept, now applied to sequential decision-making) and is the foundation of every RL algorithm, whether solved exactly (small MDPs) or approximated (large/continuous state spaces via function approximation, i.e., neural networks — Phase 5).

### Q-learning, derived
The **optimal** Q-function satisfies the Bellman optimality equation:
$$
Q^*(s,a) = E\left[R(s,a) + \gamma \max_{a'} Q^*(s',a')\right]
$$
Q-learning learns $Q^*$ iteratively from experience, without needing $P(s'|s,a)$ (model-free), via the update:
$$
Q(s,a) \leftarrow Q(s,a) + \alpha\left[\underbrace{R(s,a) + \gamma\max_{a'}Q(s',a')}_{\text{TD target}} - Q(s,a)\right]
$$
This is a **temporal difference (TD)** update — bootstrapping an estimate of $Q$ using another (more recent) estimate of $Q$, rather than waiting for a complete episode's full return, a genuinely distinct and clever idea from both pure Monte Carlo estimation (wait for the full outcome) and pure dynamic programming (require a known model).

### Policy gradient methods
Rather than learning a value function and deriving a policy from it, directly parameterize the policy $\pi_\theta(a|s)$ and optimize its parameters via gradient ascent on expected reward:
$$
\nabla_\theta J(\theta) = E_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)\right]
$$
(the **policy gradient theorem**, whose derivation uses the log-derivative trick $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$ — precisely the same trick used in variational inference and, not coincidentally, in **PPO** (Proximal Policy Optimization), the specific policy-gradient-family algorithm used in most practical RLHF implementations for LLM alignment (Phase 6).

### RLHF's reward model, connected directly to Phase 3's probability theory
In RLHF, a **reward model** is trained on human preference data (pairs of model outputs, with a human indicating which is better) using a Bradley-Terry model:
$$
P(\text{response } A \text{ preferred over } B) = \frac{e^{r(A)}}{e^{r(A)} + e^{r(B)}}
$$
trained via cross-entropy loss (Phase 3 Lesson 6) on human preference labels — the reward model itself is a supervised learning problem (Lesson 1!), and only the *subsequent* step (optimizing the language model's policy against this learned reward, typically via PPO) is genuinely reinforcement learning. This decomposition — supervised reward modeling + RL policy optimization — is exactly what Phase 6 will make fully concrete for LLM fine-tuning.

---

## 4. Algorithm — Q-Learning (fully specified, tabular case)

```
INITIALIZE: Q(s,a) arbitrarily (e.g., all zeros) for all states/actions
FOR each episode:
    s = initial state
    WHILE episode not terminated:
        CHOOSE action a using an exploration strategy (e.g., epsilon-greedy:
            with probability epsilon, choose a RANDOM action;
            otherwise choose a = argmax_a Q(s,a))
        TAKE action a, OBSERVE reward r and next state s'
        UPDATE: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        s <- s'
RETURN Q  (the learned action-value function; optimal policy = argmax_a Q(s,a) for each s)
```
**Epsilon-greedy** is the simplest exploration/exploitation balancing strategy (Section 2): a fixed probability $\epsilon$ of choosing a fully random action, decaying $\epsilon$ over training as the agent's Q-estimates become more trustworthy — simple, but a real, deliberate design choice with genuine consequences for how quickly/reliably the agent converges to a good policy.

---

## 5. Python Implementation

```python
"""reinforcement_learning_core.py — tabular Q-learning on a simple grid world"""
import numpy as np

# A simple 4x4 grid world: agent starts at (0,0), goal at (3,3), one "trap" cell
GRID_SIZE = 4
GOAL = (3, 3)
TRAP = (1, 2)
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]   # right, left, down, up


def step(state: tuple, action: tuple) -> tuple[tuple, float, bool]:
    new_state = (max(0, min(GRID_SIZE - 1, state[0] + action[0])),
                 max(0, min(GRID_SIZE - 1, state[1] + action[1])))
    if new_state == GOAL:
        return new_state, 10.0, True
    if new_state == TRAP:
        return new_state, -10.0, True
    return new_state, -0.1, False   # small negative reward per step encourages SHORT paths


def q_learning(n_episodes: int = 2000, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.2):
    Q = {}
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            Q[(i, j)] = np.zeros(len(ACTIONS))

    rng = np.random.default_rng(0)
    for episode in range(n_episodes):
        state = (0, 0)
        done = False
        while not done:
            if rng.random() < epsilon:
                action_idx = rng.integers(len(ACTIONS))            # EXPLORE
            else:
                action_idx = np.argmax(Q[state])                    # EXPLOIT

            next_state, reward, done = step(state, ACTIONS[action_idx])
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            Q[state][action_idx] += alpha * (td_target - Q[state][action_idx])   # EXACT Section 3 update
            state = next_state
    return Q


Q = q_learning()
# Extract the learned optimal policy
for i in range(GRID_SIZE):
    row = []
    for j in range(GRID_SIZE):
        best_action = np.argmax(Q[(i, j)])
        row.append(["R", "L", "D", "U"][best_action])
    print(row)
```

---

## 6. Build From Scratch

Section 5's Q-learning implementation already *is* a from-scratch build (tabular Q-learning is simple enough to implement fully, with no simplification needed). The natural additional from-scratch exercise is a **REINFORCE** (basic policy gradient) implementation, to make Section 3's policy gradient theorem concrete:

```python
import numpy as np

def softmax_policy(state_features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    logits = state_features @ theta   # theta: (n_features, n_actions)
    exp_logits = np.exp(logits - logits.max())
    return exp_logits / exp_logits.sum()

def reinforce_update(theta: np.ndarray, episode_states: list, episode_actions: list,
                       episode_rewards: list, gamma: float = 0.99, lr: float = 0.01) -> np.ndarray:
    """Monte Carlo policy gradient -- update AFTER a full episode using the ACTUAL observed return."""
    T = len(episode_rewards)
    returns = np.zeros(T)
    running_return = 0
    for t in reversed(range(T)):
        running_return = episode_rewards[t] + gamma * running_return
        returns[t] = running_return   # G_t = actual discounted return from time t onward

    for t in range(T):
        probs = softmax_policy(episode_states[t], theta)
        grad_log_pi = -probs[None, :] * episode_states[t][:, None]
        grad_log_pi[:, episode_actions[t]] += episode_states[t]   # d(log pi(a|s))/d(theta), softmax gradient
        theta += lr * grad_log_pi * returns[t]                     # EXACTLY Section 3's policy gradient theorem
    return theta
```
This directly implements $\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log\pi_\theta(a_t|s_t) \cdot G_t$ (using the actual observed return $G_t$ as a Monte Carlo estimate of $Q^\pi(s_t,a_t)$) — the historically original policy gradient algorithm (Williams, 1992), whose core mathematical idea (log-derivative trick + weighting by return) is the direct ancestor of PPO's more sophisticated, more stable RLHF-relevant update rule.

---

## 7. Library/Tool Comparison

| From scratch | Production tooling |
|---|---|
| `q_learning` (tabular) | `stable-baselines3` (DQN, PPO, etc.) — neural-network function approximation for large/continuous state spaces, far beyond tabular feasibility |
| `reinforce_update` | `stable-baselines3.PPO` — adds crucial stability improvements (clipped objective, advantage estimation) over plain REINFORCE's high variance |
| No RLHF pipeline shown | `trl` (Hugging Face's Transformer Reinforcement Learning library) — implements the full RLHF pipeline (reward model training + PPO fine-tuning) directly on top of Transformers models, the production tool for Phase 6's LLM alignment work |

---

## 8. Visual Explanations

**The RL agent-environment feedback loop:**
```
        ┌─────────────┐   action a_t    ┌─────────────┐
        │             │ ──────────────▶ │             │
        │    AGENT    │                 │ ENVIRONMENT │
        │             │ ◀────────────── │             │
        └─────────────┘  state s_{t+1}, └─────────────┘
                          reward r_t
   (Agent updates its policy/value estimates based on this ongoing feedback loop)
```

**Q-learning's temporal difference update (bootstrapping, visualized):**
```
Q(s,a) ─────────update using─────────▶  r + gamma * max Q(s', a')
  ▲                                              ▲
  │ current estimate                             │ "TD target" -- uses the NEXT state's
  │ (to be corrected)                               CURRENT (possibly still-learning) estimate
  │                                                 of Q, not the true, unknown Q* -- this
  └── the "bootstrapping" in the name                 is what makes it work WITHOUT a full episode
```

---

## 9. Practical Examples

**Simple:** implement Q-learning on the 4x4 grid world (Section 5) and visualize the learned optimal policy as arrows.
**Medium:** implement epsilon-greedy with a decaying epsilon schedule and compare convergence speed against a fixed epsilon.
**Real-world (conceptual, given this is an introduction lesson)**: read and summarize how RLHF's reward-model-plus-PPO pipeline (Section 3) is structured in a real open-source implementation (e.g., Hugging Face's `trl` library documentation), connecting each piece back to this lesson's MDP/Bellman/policy-gradient framework — direct preparation for Phase 6's fine-tuning content.

---

## 10. Real Industry Use Cases

- **RLHF for LLM alignment** (OpenAI's InstructGPT/ChatGPT, Anthropic's Constitutional AI-informed training, and virtually every major assistant model in 2026): the single most consequential real-world RL application today, directly using the reward-model + PPO structure from Section 3.
- **Recommendation systems**: many large-scale recommender systems (YouTube, Netflix) incorporate RL-inspired techniques (contextual bandits, a simplified stateless RL variant) to balance exploration (showing novel content) against exploitation (showing known-good content).
- **Robotics and autonomous systems**: classical deep RL (DQN, PPO, SAC) remains the standard approach for learning control policies in simulation before real-world deployment.
- **Game-playing systems** (AlphaGo, AlphaZero, OpenAI Five): the historically famous demonstrations of deep RL's capability, combining Q-learning/policy-gradient-family methods with deep neural network function approximation (Phase 5) and, for AlphaGo/Zero, Monte Carlo Tree Search.

---

## 11. Common Mistakes

- Confusing RL with supervised learning — there is no "correct label" per state, only a reward signal, and credit assignment (which past action caused a later reward) is a genuinely distinct, harder problem.
- Under-exploring (a purely greedy policy from the start) — the agent can converge prematurely to a mediocre policy without ever discovering a better one, a direct consequence of neglecting the exploration/exploitation tradeoff (Section 2).
- Forgetting that Q-learning's TD target bootstraps off a *still-learning* estimate — early training can be noisy/unstable, and this is expected, not necessarily a bug.
- Assuming RLHF is "pure RL" — as Section 3 makes explicit, the reward-modeling step is ordinary supervised learning (Lesson 1); only the policy-optimization step is RL, a distinction that matters for correctly attributing where in the pipeline various failure modes (reward hacking, distribution shift) actually originate.

---

## 12. Best Practices (2026)

- Use established libraries (`stable-baselines3`, `trl`) for any real RL/RLHF implementation rather than hand-rolling tabular methods beyond an educational context — production RL involves numerous stability tricks (advantage normalization, clipped objectives, reward normalization) well beyond this introductory lesson's scope.
- Always benchmark an RL agent against simple baselines (random policy, greedy-only policy) — analogous to Lesson 6's naive-forecast baseline discipline, to confirm the learned policy is genuinely adding value.
- When studying RLHF specifically (Phase 6 preview), keep the reward-modeling (supervised) and policy-optimization (RL) stages conceptually separate — most practical debugging and improvement work happens in the reward model's data quality, not the RL algorithm itself.
- Treat reward function design as a first-class, high-stakes engineering task — poorly specified rewards reliably produce "reward hacking" (the agent finding unintended ways to maximize reward that don't reflect the true intended goal), a well-documented failure mode across both classical RL and RLHF.

---

## 13. Exercises

**Easy:** Implement Q-learning on a simple 1D "line world" (agent must learn to move right to reach a goal) and verify it converges to the optimal policy.
**Medium:** Compare epsilon-greedy with a fixed epsilon vs. a decaying epsilon schedule on the Section 5 grid world, measuring episodes-to-convergence for each.
**Hard:** Implement the from-scratch REINFORCE algorithm (Section 6) on a slightly larger grid world using a linear policy over simple state features, and compare its sample efficiency (episodes needed to reach a good policy) against tabular Q-learning.
**Mathematical:** Derive the Bellman equation for $V^\pi$ from the definition of the discounted cumulative reward, using the law of total expectation (Phase 3 Lesson 3).
**Coding:** Modify the Section 5 grid world to include stochastic transitions (actions succeed only 80% of the time, otherwise a random other action occurs) and verify Q-learning still converges to a sensible (if more cautious) policy.

---

## 14. Mini Project

Build a **complete tabular RL agent comparison**: implement Q-learning and SARSA (an on-policy TD variant, differing from Q-learning's off-policy `max` in the TD target) on the same grid-world environment, compare their learned policies especially near the trap cell (SARSA tends to learn more "cautious" policies near penalty states, an instructive on-policy vs. off-policy distinction), and write a short explanation connecting the reward-shaping choices you made (the `-0.1` per-step penalty, the trap's `-10` penalty) to the resulting learned behavior — direct, hands-on intuition for the reward-design sensitivity that becomes critical in Phase 6's RLHF context.

---

## 15. Interview Preparation

- Explain the Bellman equation and why it enables a recursive, dynamic-programming-style solution to sequential decision problems.
- What is the exploration-exploitation tradeoff, and how does epsilon-greedy address it?
- Explain the difference between Q-learning (off-policy) and policy gradient methods, at a conceptual level.
- How does RLHF combine supervised learning (reward modeling) and reinforcement learning (policy optimization), and why is this decomposition useful?

---

## 16. Summary

Reinforcement learning formalizes sequential decision-making under a reward signal via the Markov Decision Process framework, with the Bellman equation's recursive structure (a dynamic-programming idea, Phase 1 Lesson 4) underlying both value-based methods (Q-learning's temporal-difference bootstrapping) and policy-based methods (the policy gradient theorem's log-derivative trick). This lesson's primary 2026 relevance is as direct preparation for Phase 6: RLHF — the technique that aligns raw pretrained language models into helpful, instruction-following assistants — decomposes cleanly into a supervised reward-modeling stage (Lesson 1's territory) followed by a genuine RL policy-optimization stage (PPO, a more stable descendant of this lesson's REINFORCE), making this introduction the conceptual bridge between classical ML and modern LLM alignment.

---

## 17. References

- Sutton, R. & Barto, A. — *Reinforcement Learning: An Introduction* (free online, the definitive RL textbook)
- Watkins, C. — "Learning from Delayed Rewards" (1989, the original Q-learning thesis)
- Williams, R. — "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992, REINFORCE)
- Ouyang et al. — "Training Language Models to Follow Instructions with Human Feedback" (2022, the InstructGPT/RLHF paper — essential direct preparation for Phase 6)
