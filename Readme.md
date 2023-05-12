# Reinforcement Learning - Gym Sandbox

## Mountain Car Problem

https://gymnasium.farama.org/environments/classic_control/mountain_car/

## Temporal Difference Learning (TD Learning)

Bellman equation:

![bellman_equation](assets/bellman_equation.png)


$$
Q_{k+1}(s,a)
$$

is the new Q_value of the considered state, the one to update
 

$$
Q_k(s,a)
$$

is the actual Q_value of the considered state

$$
max_{a'}Q_k(s', a')
$$

is the maximum of Q_value along its actions, of a chosen next state

