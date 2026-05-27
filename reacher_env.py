# src/reacher_env.py
import numpy as np
from kinematics import ArmKinematics

class CollaborativeKinematicReacherEnv:
    def __init__(self, alpha=0.4):
        """
        Multi-Agent Simulation Framework mapping human-robot collaborative motor tasks.
        alpha: Reward weight for maximizing workspace manipulability.
        
        Agent 1 ('robot'): Controls the velocity of the shoulder joint (theta1_dot).
        Agent 2 ('human'): Controls the velocity of the elbow joint (theta2_dot).
        """
        self.kine = ArmKinematics(l1=1.0, l2=1.0)
        
        # State vector: [theta1, theta2, theta1_dot, theta2_dot]
        self.state = np.zeros(4, dtype=np.float32)
        
        # Fixed tracking target in Cartesian Space
        self.target_pos = np.array([0.7, 0.7], dtype=np.float32)
        
        self.dt = 0.05      # Simulation time step
        self.alpha = alpha  # Singularity avoidance penalty scale
        
        # Define explicit agent IDs
        self.agents = ["robot", "human"]

    def _get_obs(self, mu, ee_pos):
        """Generates localized multi-agent observations."""
        # Both agents see the full state, end-effector position, target position, and manipulability
        shared_obs = np.hstack([self.state, ee_pos, self.target_pos, mu], dtype=np.float32)
        return {
            "robot": shared_obs,
            "human": shared_obs
        }

    def step(self, action_dict):
        """
        Processes collaborative multi-agent state changes.
        action_dict: Dictionary containing continuous 1D actions for each agent:
                     {"robot": continuous_value, "human": continuous_value}
        """
        # 1. Unpack current joint geometry
        theta1, theta2 = self.state[0], self.state[1]
        
        # 2. Extract and bound individual joint velocities outputted by the independent policies
        # Clip to protect the physical limits of the actuators
        q_dot_robot = np.clip(action_dict["robot"], -1.5, 1.5)
        q_dot_human = np.clip(action_dict["human"], -1.5, 1.5)
        
        # Consolidate the independent actions into a unified joint velocity vector
        q_dot_input = np.array([q_dot_robot, q_dot_human], dtype=np.float32)
        
        # 3. Compute current task space velocity and check structural limitations
        J = self.kine.compute_jacobian(theta1, theta2)
        mu = self.kine.compute_manipulability(J)
        
        # Pass through the inverse engine to ensure DLS regulates any singular commands
        # If actions cause a lock, the kinematic engine steps in to damp velocities safely
        x_dot_actual = np.dot(J, q_dot_input)
        q_dot_safe, _ = self.kine.differential_inverse_kinematics(theta1, theta2, x_dot_actual)
        
        # 4. State Integration (Euler Forward Method)
        self.state[0] += q_dot_safe[0] * self.dt
        self.state[1] += q_dot_safe[1] * self.dt
        self.state[2], self.state[3] = q_dot_safe[0], q_dot_safe[1] # Update internal velocity state
        
        # 5. Extract new physical coordinates via Forward Kinematics
        ee_pos = self.kine.forward_kinematics(self.state[0], self.state[1])
        
        # 6. Collaborative Reward Engineering: Shared tracking error + team geometry incentive
        distance = np.linalg.norm(ee_pos - self.target_pos)
        shared_reward = -(distance ** 2) + (self.alpha * mu)
        
        rewards = {
            "robot": shared_reward,
            "human": shared_reward
        }
        
        # Modern multi-agent termination condition
        done_status = bool(distance < 0.05)
        terminated = {"robot": done_status, "human": done_status}
        truncated = {"robot": False, "human": False}
        
        # Generate new decentralized observation sets
        obs_dict = self._get_obs(mu, ee_pos)
        
        info_dict = {
            "ee_position": ee_pos,
            "manipulability": mu,
            "tracking_error": distance
        }
        
        return obs_dict, rewards, terminated, truncated, info_dict

    def reset(self):
        """Resets dyadic system to a stable configuration away from alignment limits."""
        # Intentionally initialize away from structural locking positions (0.0, 0.0)
        self.state = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        
        ee_pos = self.kine.forward_kinematics(self.state[0], self.state[1])
        J = self.kine.compute_jacobian(self.state[0], self.state[1])
        mu = self.kine.compute_manipulability(J)
        
        return self._get_obs(mu, ee_pos), {}

# --- Mock Multi-Agent Validation Pipeline ---
if __name__ == "__main__":
    env = CollaborativeKinematicReacherEnv()
    obs_dict, info = env.reset()
    
    print("=================== MARL PIPELINE VERIFICATION ===================")
    print(f"Initial Robot View:\n{obs_dict['robot']}\n")
    print(f"Initial Human View:\n{obs_dict['human']}\n")
    
    # Simulating cooperative simultaneous joint control inputs from two un-trained network policies
    mock_marl_actions = {
        "robot": np.array(0.5, dtype=np.float32),  # Commanding shoulder velocity
        "human": np.array(-0.3, dtype=np.float32)  # Commanding elbow velocity
    }
    
    next_obs, rewards, terminated, truncated, step_info = env.step(mock_marl_actions)
    
    print(f"MARL Input Logged:          {mock_marl_actions}")
    print(f"Calculated End-Effector:    {np.round(step_info['ee_position'], 4)}")
    print(f"Manipulability Score (μ):  {np.round(step_info['manipulability'], 4)}")
    print(f"Team Rewards Returned:      {rewards}")
    print(f"Termination Status:         {terminated}")
    print("==================================================================")