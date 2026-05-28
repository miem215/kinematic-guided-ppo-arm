# Kinematic-Guided PPO Arm

A multi-agent reinforcement learning (MARL) simulation framework that maps human-robot collabration using a 2-DoF planar manipulator. The environment embeds robot kinematics directly into the control loop, enabling independent learning policies to cooperate while safely steering clear of kinamtic singularities.

## Overview

This project simulates a **2-Degree-of-Freedom (2-DoF) Planar Arm** co-managed by two separate independent agents. It bounds and transforms joint velocities via **Differential Inverse Kinematics** and **Damped Least Squares (DLS)** fallback configurations. This ensures that even when untrained RL agents command extreme or dangerous actions, the underlying math safeguards the physical limitations of the arm.

---

## Key Components & Architecture

### 1. Kinematics Engine (`kinematics.py`)
Handles the mathematical mapping between the joint space and Cartesian task space:
* **Forward Kinematics:** Maps joint angles $(\theta_1, \theta_2)$ to Cartesian end-effector coordinates $(x, y)$.
* **Analytical Jacobian:** Computes the $2\times2$ Jacobian Matrix $J(q)$ mapping joint velocities to end-effector velocities.
* **Yoshikawa Manipulability Index:** Evaluates structural distance from kinematic singularities using $\mu = |\det(J)|$.
* **Singularity-Robust Inversion:** Implements an adaptive **Damped Least Squares (DLS)** fallback mechanism. When the arm approaches a singularity ($\mu < 10^{-3}$), damping coefficients smoothly kick in to bound and regulate unsafe velocity spikes.

### 2. Cooperative Reacher Environment (`reacher_env.py`)
A multi-agent environment built to support shower and elbow joints coordination:
* **Agent 1 (`robot`):** Controls the velocity of the shoulder joint $\dot{\theta}_1$.
* **Agent 2 (`human`):** Controls the velocity of the Elbow joint $\dot{\theta}_2$.
* **Observation Space:** Both agents receive shared localized observations containing: `[theta1, theta2, theta1_dot, theta2_dot, ee_x, ee_y, target_x, target_y, manipulability]`.
* **Collaborative Reward:** The team shares a dense reward structured to minimize tracking error while maximizing structural manipulability to actively avoid locking:
  $$\text{Reward} = -(\text{distance}^2) + (\alpha \cdot \mu)$$

---

## Mathematics Behind the Guidance

### Yoshikawa Manipulability
To avoid structural locking where the arm is completely straight or folded into itself, the environment scores the arm's posture dynamically:
$$\mu(q) = \sqrt{\det(J(q)J^T(q))}$$

### Damped Least Squares (Pseudo-Inverse)
When the manipulability drops below the threshold, clean matrix inversion breaks down. The environment automatically shifts to a robust pseudoinverse:
$$J^* = J^T (JJ^T + \lambda^2 I)^{-1}$$
where the damping factor $\lambda$ scales adaptively based on proximity to the singular boundary.

---
