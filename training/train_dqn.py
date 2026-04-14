"""
training/train_dqn.py
────────────────────
The DQN train

The RL agent's job is to get as close to the full-information baseline model as possible 
with the constraint of feature acquisition having associated costs.
"""

import os
import sys
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate import evaluate_dqn

def train_dqn(
    env,
    agent,
    num_episodes=500,       
    max_steps=19,
    eval_every=25,
    save_path = os.path.join(config.CHECKPOINT_DIR, "predictor_dqn.pt")
):
    """
    Main training loop for dqn_agent.

    Parameters
    ----------
    env : environment instance
    agent : dqn_agent
    num_episodes : total training episodes (molecules to train on)
    max_steps : max steps per episode, default is 19 b/c there are only 19 properties
    eval_every : evaluate every N episodes
    save_path : checkpoint file path

    Returns
    -------
    reward_history : list of rewards for each episode
    loss_history : list of losses for each episode
    """

    reward_history = []
    loss_history = []

    for episode in range(0, num_episodes):
        #Reset episode
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_loss = []

        #Run one episode, acquiring properties until all 19 are acquired or max_steps is hit
        for i in range(max_steps):
            #Get the legal properties as a mask for the agent to choose from
            legal_mask = env.legal_action_mask()

            #Select action and acquire 1 property determined from action
            action = agent.select_action(state, legal_mask)
            next_state, reward, done, info = env.step(action)

            #Store transition
            agent.store(state, action, reward, next_state, done)

            #Learn 
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            #Move to next_state and update reward
            state = next_state
            episode_reward += reward

            if done:
                break

        #Tracking avg_loss
        avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0.0

        reward_history.append(episode_reward)
        loss_history.append(avg_loss)

        print(
            f"Current Episode: {episode:4d} | "     #NB: starts at 0
            f"Reward: {episode_reward:8.3f} | "
            f"Loss: {avg_loss:8.4f} | "
            f"Eps: {agent.epsilon:6.3f} | "
            f"Buffer: {len(agent.buffer)}"
        )

        #Periodic evaluation as determined by eval_every
        if episode % eval_every == 0:
            eval_reward = evaluate_dqn(env, agent, episodes = 5)
            print(f"   Eval Avg Reward: {eval_reward:.3f}")

            agent.save(save_path)

    return reward_history, loss_history
