# src/reacher_env.py
import numpy as np
from kinematics import ArmKinematics

class KinematicReacherEnv:
    def __init__(self, alpha=0.4):
        """
        Simulation framework wrapping physical system integration and the RL reward structure.
        alpha: Reward weight for maximizing workspace manipulability.
        """
        # Instantiate our mathematical kinematics model
        self.kine = ArmKinematics(l1=1.0, l2=1.0)
        
        # State vector: [theta1, theta2, theta1_dot, theta2_dot]
        self.state = np.zeros(4, dtype=np.float32)
        
        # Fixed tracking target in Cartesian Space
        self.target_pos = np.array([0.7, 0.7], dtype=np.float32)
        
        self.dt = 0.05      # Simulation time step
        self.alpha = alpha  # Singularity avoidance penalty scale

    def step(self, action):
        """
        Processes the environment state change based on policy network decisions.
        action: Continuous 2D task-space velocity vector [x_dot, y_dot] output by RL.
        """
        # 1. Unpack geometry
        theta1, theta2 = self.state[0], self.state[1]
        
        # 2. Bound the incoming continuous command vector to physical limitations
        x_dot_ref = np.clip(action, -1.2, 1.2)
        
        # 3. Pass task commands through the Differential Inverse Kinematics engine
        q_dot, mu = self.kine.differential_inverse_kinematics(theta1, theta2, x_dot_ref)
        
        # 4. State Integration (Euler Forward Method)
        self.state[0] += q_dot[0] * self.dt
        self.state[1] += q_dot[1] * self.dt
        self.state[2], self.state[3] = q_dot[0], q_dot[1] # Update internal velocity state
        
        # 5. Extract new physical coordinates via Forward Kinematics
        ee_pos = self.kine.forward_kinematics(self.state[0], self.state[1])
        
        # 6. Reward Engineering: Quadratic tracking loss + Geometry incentive
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = -(distance ** 2) + (self.alpha * mu)
        
        # Done condition (Success criteria if end-effector tracks within 5cm)
        done = bool(distance < 0.05)
        
        # Consolidate observation space for network evaluation
        obs = np.hstack([self.state, ee_pos, self.target_pos, mu], dtype=np.float32)
        
        info = {
            "ee_position": ee_pos,
            "manipulability": mu,
            "tracking_error": distance
        }
        
        return obs, reward, done, info

    def reset(self):
        """Resets system to a stable, non-singular configuration away from alignment limits."""
        # Intentionally initialize away from structural locking positions (0.0, 0.0)
        self.state = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        
        ee_pos = self.kine.forward_kinematics(self.state[0], self.state[1])
        J = self.kine.compute_jacobian(self.state[0], self.state[1])
        mu = self.kine.compute_manipulability(J)
        
        obs = np.hstack([self.state, ee_pos, self.target_pos, mu], dtype=np.float32)
        return obs

# --- Mock Validation Pipeline ---
if __name__ == "__main__":
    env = KinematicReacherEnv()
    observation = env.reset()
    
    print("=================== PIPELINE VERIFICATION ===================")
    print(f"Initial State Observation Map:\n{observation}\n")
    
    # Simulating a dynamic diagonal step command from an un-trained Actor Policy
    mock_policy_action = np.array([0.8, -0.4], dtype=np.float32)
    next_obs, step_reward, done_status, step_info = env.step(mock_policy_action)
    
    print(f"RL Command Received:        {mock_policy_action}")
    print(f"Calculated End-Effector:    {np.round(step_info['ee_position'], 4)}")
    print(f"Manipulability Score (μ):  {np.round(step_info['manipulability'], 4)}")
    print(f"Computed Step Reward:       {np.round(step_reward, 4)}")
    print("=============================================================")