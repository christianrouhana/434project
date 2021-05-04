# Partially adapted from: https://keras.io/examples/rl/actor_critic_cartpole/
# & https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
import gym
import keras
from keras import layers
import tensorflow as tf
import numpy as np
import sys
import os
from os.path import isfile
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

best_hidden = 0
best_lr = 0
best_episodes = 500
iteration = 0
for HIDDEN in [64, 128, 256, 512, 1024]:
	for LR in [0.001, 0.003, 0.01, 0.1]:
			os.system('clear')
			print('Now testing learning rate {}, hidden layer val {}'.format(LR, HIDDEN))
			seed = 42
			n_actions = 2
			n_inputs = 4
			n_hidden = HIDDEN
			lr = LR
			gamma = 0.99
			MAX_STEPS = 10000 #max steps per episode
			eps = np.finfo(np.float32).eps.item()
			states, action_prob_grads, rewards, action_probs = [], [], [], []
			reward_sum = 0
			episode_number = 0

			#common layers
			inputs = layers.Input(shape = (n_inputs, ))
			l1 = layers.Dense(n_hidden)(inputs)
			l2 = tfa.activations.mish(l1)
			l3 = layers.Dense(n_hidden)(l2)
			common = tfa.activations.mish(l3)

			#create actor and critic model
			action = layers.Dense(n_actions, activation = 'softmax')(common)
			critic = layers.Dense(1)(common)
			model = keras.Model(inputs=inputs, outputs=[action, critic])
			optimizer = keras.optimizers.Adam(learning_rate=lr)
			loss = keras.losses.Huber()

			action_probs_history = []
			critic_value_history = []
			rewards_history = []
			episode_count = 0
			running_reward = 0
			reward_plot = []

			env = gym.make('CartPole-v0')
			env.seed(seed)
			while True:
			    state = env.reset()
			    episode_reward = 0

			    with tf.GradientTape() as tape:
			        for step in range(1, MAX_STEPS):
			            state = tf.convert_to_tensor(state)
			            state = tf.expand_dims(state, 0)

			            action_probs, critic_value = model(state)
			            critic_value_history.append(critic_value[0,0])

			            action = np.random.choice(n_actions, p = np.squeeze(action_probs))
			            action_probs_history.append(tf.math.log(action_probs[0, action]))

			            state, reward, done, _ = env.step(action)
			            rewards_history.append(reward)
			            episode_reward += reward

			            if done: break

			        running_reward = 0.05 * episode_reward + (1-0.05) * running_reward
			        reward_plot.append(running_reward)
			        returns = []
			        discounted_sum = 0
			        for r in rewards_history[::-1]:
			            discounted_sum = r + gamma * discounted_sum
			            returns.insert(0, discounted_sum)

			        returns = np.array(returns)
			        returns = (returns - np.mean(returns))/(np.std(returns)+eps)
			        returns = returns.tolist()

			        history = zip(action_probs_history, critic_value_history, returns)
			        actor_losses = []
			        critic_losses = []
			        for log_prob, value, ret in history:
			            diff = ret - value
			            actor_losses.append(-log_prob * diff)
			            #update critic
			            critic_losses.append(
			                loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
			            )

			        loss_value = sum(actor_losses) + sum(critic_losses)
			        grads = tape.gradient(loss_value, model.trainable_variables)
			        optimizer.apply_gradients(zip(grads, model.trainable_variables))

			        action_probs_history.clear()
			        critic_value_history.clear()
			        rewards_history.clear()

			        episode_count += 1
			        if episode_count % 10 == 0:
			            print("reward: {} at episode {}".format(running_reward, episode_count))
			        if running_reward > 195:
			            print("Solved at episode {}".format(episode_count))
			            break
			        if episode_count == 399:
			        	print('Took too long, scrapping')
			        	break
			env.close()
			print('iteration {} complete'.format(iteration))
			iteration += 1
			if episode_count < best_episodes:
				best_episodes = episode_count
				best_lr = LR
				best_hidden = HIDDEN

os.system('clear')
print('MISH HYPERPARAM SWEEP RESULTS')
print('Best learning rate: {}'.format(best_lr))
print('Best hidden: {}'.format(best_hidden))
print('Best convergence: {} Episodes'.format(best_episodes))