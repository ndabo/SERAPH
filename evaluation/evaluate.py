"""
evaluate.py
────────────────────
Evaluation of DQN performance
"""

def evaluate_dqn(env, agent, episodes=5):
    """
    Greedy evaluation (no exploration)

    Parameters
    ----------
    env : environment instance
    agent : dqn_agent

    Returns
    -------
    mean_scores : the mean of scores (total rewards per episode), duh
    """
    scores = []

    for i in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            legal_mask = env.legal_action_mask()
            action = agent.select_action(state, legal_mask, force_greedy = True)

            state, reward, done, info = env.step(action)
            total_reward += reward

        scores.append(total_reward)

    return sum(scores) / len(scores)