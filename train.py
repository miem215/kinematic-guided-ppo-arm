
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from reacher_env import CollaborativeKinematicReacherEnv
from models import ActorCritic, PPORolloutBuffer

class IPPOAgent:
    def __init__(self, state_dim=9, action_dim=1, lr=3e-4, gamma=0.99, K_epochs=5, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = PPORolloutBuffer()
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            dist, _ = self.policy_old(state_t)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action.numpy())
        self.buffer.log_probs.append(action_log_prob.item())
        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalize targets
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensors
        old_states = torch.FloatTensor(np.array(self.buffer.states))
        old_actions = torch.FloatTensor(np.array(self.buffer.actions))
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs))

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            dist, state_values = self.policy(old_states)
            state_values = torch.squeeze(state_values)
            log_probs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            
            # PPO Core Ratio Calculation
            ratios = torch.exp(log_probs - old_log_probs)

            # Surrogate Losses
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Combined Loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Gradient Descent Step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

def train_cooperative_system():
    env = CollaborativeKinematicReacherEnv(alpha=0.4)
    
    # Instantiate Independent Agents matching the env IDs
    agents = {
        "robot": IPPOAgent(state_dim=9, action_dim=1),
        "human": IPPOAgent(state_dim=9, action_dim=1)
    }
    
    max_episodes = 1000
    print("=================== TRAINING INDEPENDENT PPO POLICIES ===================")
    
    for episode in range(1, max_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Each agent evaluates its shared observation matrix independently
            action_dict = {
                "robot": np.array(agents["robot"].select_action(obs["robot"]), dtype=np.float32),
                "human": np.array(agents["human"].select_action(obs["human"]), dtype=np.float32)
            }
            
            # Env step handles clipping and kinematic constraints automatically
            next_obs, rewards, terminated, truncated, step_info = env.step(action_dict)
            done = any(terminated.values())
            
            # Cache transitions into respective agent buffers
            for agent_id in env.agents:
                agents[agent_id].buffer.rewards.append(rewards[agent_id])
                agents[agent_id].buffer.is_terminals.append(done)
                
            obs = next_obs
            episode_reward += rewards["robot"] # Rewards are shared/cooperative
            
        # Run optimization loops after closing out the episode trajectory
        for agent_id in env.agents:
            agents[agent_id].update()
            
        if episode % 25 == 0 or episode == 1:
            print(f"Ep {episode:4d} | Team Reward: {episode_reward:7.2f} | Error: {step_info['tracking_error']:.3f} | Manipulability (μ): {step_info['manipulability']:.3f}")

if __name__ == "__main__":
    train_cooperative_system()