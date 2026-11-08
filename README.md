# Reinforcement Learning Algorithms in a Stochastic Gridworld

A Python implementation and comparison of model-based and model-free reinforcement learning algorithms in a custom stochastic Gridworld environment.

The project explores three approaches to finding an optimal policy:

* Q-Value Iteration
* Q-Learning
* REINFORCE (Monte Carlo Policy Gradient)

1. Environment

A custom 2 × 5 Gridworld environment is implemented using Python.

The environment consists of:

* 10 states arranged in two rows and five columns.
* 7 non-terminal states.
* 3 terminal states: B1, B3, and B5.
* 4 possible actions: Up, Down, Left, and Right.

Rewards

State	Reward
B1	-1
B3	+1
B5	-1
Non-terminal states	-0.04 per step

The transition model is stochastic. The intended action is executed with probability 0.6, while the remaining probability is distributed among alternative movements and no movement.

The environment is implemented through a custom GridEnv class with reset() and step() methods.

2. Q-Value Iteration

Q-Value Iteration is a model-based approach that assumes the transition probabilities and reward function are known.

The action-value function is updated using:

[
Q(s,a)=\mathbb{E}[r]+\gamma\mathbb{E}\left[\max_{a’}Q(s’,a’)\right]
]

Expected values are calculated using the complete transition distribution.

Experimental Setup

* Discount factor: γ = 0.9
* Convergence tolerance: 10⁻¹⁰
* Maximum iterations: 1,000

The algorithm converged after 45 iterations.

The resulting optimal policy is:

State	Optimal Action
A1	Right
A2	Right
A3	Down
A4	Left
A5	Left
B2	Right
B4	Left

The optimal policy guides the agent toward the positive-reward terminal state B3 while avoiding B1 and B5.

3. Q-Learning

Q-Learning is a model-free reinforcement learning algorithm that learns directly through interaction with the environment.

The implementation uses the temporal-difference update:

[
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+
\alpha\left[r_t+\gamma\max_{a’}Q(s_{t+1},a’)-Q(s_t,a_t)\right]
]

Exploration Strategy

An ε-greedy strategy is used:

* With probability ε, a random action is selected.
* Otherwise, the action with the highest Q-value is selected.

Experimental Setup

* Number of episodes: 20,000
* Exploration probability: ε = 0.1
* Discount factor: γ = 0.9
* Decaying learning rate:

[
\alpha_e=\frac{\ln(e+1)}{e+1}
]

Results

The learned greedy policy matched the optimal policy from Q-Value Iteration in all 7 non-terminal states.

The maximum difference between learned and reference state values was:

[
\max_s |V_{\text{learned}}(s)-V^*(s)|=0.0107
]

A comparison between learning-rate strategies showed:

Learning Rate	Maximum Value Error
Decaying	0.0107
Constant (α = 0.1)	0.3276

The decaying learning rate provided substantially more accurate value estimates.

The learned policy was also evaluated over 10 test episodes, achieving an average undiscounted reward of 0.94.

An additional experiment with γ = 0.999 showed that the learned policy matched the reference policy in 6 out of 7 non-terminal states.

4. REINFORCE — Policy Gradient

REINFORCE is a model-free policy-gradient algorithm that directly optimizes a parameterized stochastic policy.

The policy is represented using a softmax distribution over action preferences (logits) for each non-terminal state:

[
\pi(a|s)=\operatorname{softmax}(\theta(s))[a]
]

The policy parameters are updated after each episode using the discounted return:

[
G_t=r_t+\gamma G_{t+1}
]

The update follows:

[
\theta(s_t,\cdot)\leftarrow
\theta(s_t,\cdot)+
\alpha G_t\nabla_\theta\log\pi(a_t|s_t)
]

Experimental Setup

* Discount factor: γ = 0.9
* Number of episodes: 8,000
* Learning rate: α = 0.05
* Softmax action selection

Results

The learned policy matched the reference optimal policy in all 7 non-terminal states.

The average reward increased from approximately -0.34 during the first 200 episodes to approximately 0.69 during the final 200 episodes.

The learned action probabilities became strongly concentrated around the optimal actions, indicating convergence toward a nearly deterministic policy.

5. Algorithm Comparison

Algorithm	Approach	Uses Environment Model?
Q-Value Iteration	Dynamic Programming	Yes
Q-Learning	Temporal-Difference Learning	No
REINFORCE	Monte Carlo Policy Gradient	No

The experiments demonstrate three different approaches to solving a Markov Decision Process.

* Q-Value Iteration provides a reference optimal policy when the environment model is known.
* Q-Learning learns action values through interaction and exploration.
* REINFORCE directly optimizes the policy using episode returns and gradient updates.

Technologies

* Python
* NumPy
* Matplotlib
* Jupyter Notebook
* Reinforcement Learning
* Markov Decision Processes

Repository Structure

reinforcement-learning-algorithms/
│
├── notebooks/
│   └── reinforcement_learning_gridworld.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md

How to Run

Clone the repository:

git clone https://github.com/cicvaricjelena2/reinforcement-learning-algorithms.git

Install dependencies:

pip install -r requirements.txt

Launch Jupyter Notebook:

jupyter notebook

Open reinforcement_learning_gridworld.ipynb and run the cells.

Conclusion

This project demonstrates the implementation and comparison of model-based and model-free reinforcement learning methods in a stochastic environment.

By comparing Q-Value Iteration, Q-Learning, and REINFORCE, the project illustrates how an agent can learn an optimal decision-making strategy using either a known environment model or experience collected through interaction.
