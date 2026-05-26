# src/kinematics.py
import numpy as np

class ArmKinematics:
    def __init__(self, l1=1.0, l2=1.0):
        """
        Handles the geometric and differential mapping of a 2-DoF planar manipulator.
        l1: Length of the proximal link (shoulder to elbow)
        l2: Length of the distal link (elbow to end-effector)
        """
        self.l1 = l1
        self.l2 = l2

    def forward_kinematics(self, theta1, theta2):
        """Maps Joint Space coordinates (angles) to Cartesian Task Space coordinates (x, y)."""
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return np.array([x, y], dtype=np.float32)

    def compute_jacobian(self, theta1, theta2):
        """
        Computes the analytical 2x2 Jacobian Matrix: J(q) = dX/dq
        Maps joint velocities to end-effector Cartesian velocities.
        """
        J = np.array([
            [-self.l1 * np.sin(theta1) - self.l2 * np.sin(theta1 + theta2), -self.l2 * np.sin(theta1 + theta2)],
            [ self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2),  self.l2 * np.cos(theta1 + theta2)]
        ], dtype=np.float32)
        return J

    def compute_manipulability(self, J):
        """
        Computes the Yoshikawa Manipulability Index.
        Measures structural distance from kinematic singularities.
        """
        # For a square Jacobian, this simplifies to |det(J)|
        return np.sqrt(np.maximum(0.0, np.linalg.det(np.dot(J, J.T))))

    def differential_inverse_kinematics(self, theta1, theta2, x_dot_ref):
        """
        Bridges task-space commands to joint space execution via singularity robust inversion.
        Uses Damped Least Squares (Pseudo-Inverse) when approaching structural limits.
        """
        J = self.compute_jacobian(theta1, theta2)
        mu = self.compute_manipulability(J)
        
        # Singularity avoidance threshold
        if mu > 1e-3:
            # Deterministic, clean matrix inversion away from limits
            q_dot = np.dot(np.linalg.inv(J), x_dot_ref)
        else:
            # Damped Least Squares fallback to bound joint velocities near singularity boundaries
            # q_dot = J^T * (J * J^T + lambda^2 * I)^-1
            damping = 0.02
            J_jt = np.dot(J, J.T) + (damping ** 2) * np.eye(2)
            q_dot = np.dot(J.T, np.dot(np.linalg.inv(J_jt), x_dot_ref))
            
        return q_dot, mu