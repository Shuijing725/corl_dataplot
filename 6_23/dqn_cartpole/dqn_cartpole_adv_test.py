#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 19:47:14 2017

@author: anay
"""

import pickle

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import time
import os

for j in range(10):
	ENV_NAME = 'CartPole-v0'
	#ENV_NAME ='FrozenLake-v0'


	# Get the environment and extract the number of actions.
	env = gym.make(ENV_NAME)
	np.random.seed()
	env.seed()
	nb_actions = env.action_space.n

	# Next, we build a very simple model.
	model = Sequential()
	model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))
	print(model.summary())

	# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
	# even the metrics!
	memory = SequentialMemory(limit=50000, window_length=1)
	policy = BoltzmannQPolicy()
	dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
	               target_model_update=1e-2, policy=policy)
	dqn.compile(Adam(lr=1e-3), metrics=['mae'])

	# Okay, now it's time to learn something! We visualize the training here for show, but this
	# slows down training quite a lot. You can always safely abort the training prematurely using
	# Ctrl + C.
	dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

	# After training is done, we save the final weights.
	dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

	nb_episodes = 50
	# mag_adv = 0.0
	samp_time = 20

	mag_adv = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]

	if not os.path.exists('result_data'):
	    os.makedirs('result_data')

	for i in mag_adv: 
		if not os.path.exists("result_data/adv_mag="+ str(i)):
	    		os.makedirs("result_data/adv_mag="+ str(i))
		print("")
		print("-----------------------------------")
		print("adv_mag = ", i, 'iteration #', j)
		# Finally, evaluate our algorithm for 5 episodes.
		a_van,rewardList_van=dqn.test(env, nb_episodes=nb_episodes, visualize=False)
		a_samp,rewardList_samp=dqn.testAdv_samp(env, nb_episodes=nb_episodes, visualize=False, mag_adv = i, prob = 1., samp_time =samp_time)
		a_sgd,rewardList_sgd=dqn.testAdv_sgd(env, nb_episodes=nb_episodes, visualize=False, mag_adv = i, prob = 1., samp_time = samp_time)
		a_grad,rewardList_grad=dqn.testAdv_grad(env, nb_episodes=nb_episodes, visualize=False, mag_adv = i, prob = 1., samp_time =samp_time)
		a_pa,rewardList_pa=dqn.testAdv_grad_pa(env, nb_episodes=nb_episodes, visualize=False, mag_adv = i, prob = 1., samp_time = samp_time)
		print np.mean(np.array(rewardList_van))
		print np.mean(np.array(rewardList_samp))
		print np.mean(np.array(rewardList_grad))
		print np.mean(np.array(rewardList_pa))
		#print np.mean(np.array(rewardList_sgd))

		timestr = time.strftime("%Y-%m-%d %H:%M:%S")
		#print np.mean(np.array(rewardList_sgd))
		#store={"reward_list": rewardList}

		store = {"rewardList_van": rewardList_van,
		         "rewardList_samp": rewardList_samp,
		         "rewardList_grad": rewardList_grad,
		         "rewardList_pa": rewardList_pa,
		         "rewardList_sgd": rewardList_sgd
		         }
		#
		#with open('reward_adversary.pickle','wb') as f:
		#    pickle.dump(store, f,pickle.HIGHEST_PROTOCOL)
		path = 'result_data/adv_mag=' + str(i) + '/adv_test_mag_{}_iteration_{}.pickle'.format(i, j)
		with open(path,'wb') as f:
		    pickle.dump(store, f,pickle.HIGHEST_PROTOCOL)
