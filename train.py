import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from reacher_env import CollaborativeKinematicReacherEnv
from models import ActorCritic, PPORolloutBuffer

class IPPOAgent:
    def __init__(self, state_dim=9, action_dim=1, lr=1e-4, gamma=0.99, K_epochs=3, eps_clip=0.15):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = PPORolloutBuffer()
        self.MseLoss = nn.MSELoss()

    def select_action(self, norm_state):
        """Expects a normalized NumPy array observation."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(norm_state)
            dist, value = self.policy_old(state_tensor)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        # CRITICAL: Append the NORMALIZED state to the buffer
        self.buffer.states.append(norm_state)
        self.buffer.actions.append(action.numpy().flatten())
        self.buffer.log_probs.append(action_log_prob.item())
        
        return action.numpy().flatten()

    def update(self):
        # Convert buffer lists to tensors for batch gradient descent
        old_states = torch.FloatTensor(np.array(self.buffer.states))
        old_actions = torch.FloatTensor(np.array(self.buffer.actions))
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs))
        
        # Monte Carlo estimation of target returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Standard PPO Return Normalization
        returns = torch.FloatTensor(rewards)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # PPO Optimization Loop over K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and states using current policy network distributions
            dist, state_values = self.policy(old_states)
            state_values = torch.squeeze(state_values)
            
            # Match the shape for log probabilities evaluation
            log_probs = dist.log_prob(old_actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            
            # Calculate importance sampling ratios: r(theta) = pi(a|s) / pi_old(a|s)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate policy losses
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            
            # Total Loss combining Policy Loss, Critic MSE Value Loss, and Entropy Bonus
            # Note: We keep an entropy coefficient around 0.01 to ensure exploration doesn't freeze
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy
            
            # Gradient Descent Step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Synchronize parameters: Copy new weights to old policy container
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear transaction log arrays for the next episode rollout iteration
        self.buffer.clear()

class RunningStats:
    """Tracks running mean and variance for online observation normalization."""
    def __init__(self, dim):
        self.mean = np.zeros(dim, dtype=np.float32)
        self.var = np.ones(dim, dtype=np.float32)
        self.count = 1e-4  # Prevents division by zero on step one

    def update(self, x):
        self.count += 1
        old_mean = self.mean.copy()
        # Welford's algorithm for computing running variance stably
        self.mean += (x - old_mean) / self.count
        self.var += (x - old_mean) * (x - self.mean)

    def normalize(self, x):
        std = np.sqrt(self.var / self.count)
        return (x - self.mean) / (std + 1e-8)

def train_cooperative_system():
    env = CollaborativeKinematicReacherEnv(alpha=0.02)  # Balanced weight
    obs_dim = 9
    
    # This dictionary relies directly on the class defined above!
    stats = {
        "robot": RunningStats(dim=obs_dim),
        "human": RunningStats(dim=obs_dim)
    }
    
    agents = {
        "robot": IPPOAgent(state_dim=obs_dim, action_dim=1),
        "human": IPPOAgent(state_dim=obs_dim, action_dim=1)
    }
    max_episodes = 5000
    max_steps_per_episode = 200
    print("=================== TRAINING INDEPENDENT PPO POLICIES ===================")
    
    for episode in range(1, max_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps_per_episode:
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
            steps += 1
            
        # Run optimization loops after closing out the episode trajectory
        for agent_id in env.agents:
            agents[agent_id].update()
            
        if episode % 25 == 0 or episode == 1:
            print(f"Ep {episode:4d} | Team Reward: {episode_reward:7.2f} | Error: {step_info['tracking_error']:.3f} | Manipulability (μ): {step_info['manipulability']:.3f}")

if __name__ == "__main__":
    train_cooperative_system()