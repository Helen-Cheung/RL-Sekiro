'''
Description: 
Version: 2.0
Autor: Zhang
Date: 2021-11-29 09:29:29
LastEditors: Zhang
LastEditTime: 2021-12-08 23:30:54
'''

#from cv2 import getOptimalNewCameraMatrix, threshold
import torch
from torch.autograd import Variable
import sys
import os
import gym.spaces
import itertools
import numpy as np
import random
from collections import namedtuple
from utils.replay_buffer import *
from utils.schedules import *
#from utils.gym_setup import *
#from logger import Logger
import time

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Set the logger
#logger = Logger('./logs')
def to_np(x):
    return x.data.cpu().numpy() 

def dqn_learning(env,
          env_id,
          q_func,
          optimizer_spec,
          exploration=LinearSchedule(1000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10,
          double_dqn=False,
          dueling_dqn=False,
          checkpoint = 0):


    ################
    #  BUILD MODEL #
    ################


    img_w = env.width
    img_h = env.height
    img_c = 3
    input_shape = (img_h, img_w, frame_history_len * img_c)
    in_channels = input_shape[2]

    num_actions = env.action_dim

    # define Q target and Q
    Q = q_func(num_classes=num_actions, net="B0", pretrained=False).type(dtype)
    Q_target = q_func(num_classes=num_actions, net="B0", pretrained=False).type(dtype)

    # load checkpoint
    if checkpoint != 0:
        add_str = ''
        if (double_dqn):
            add_str = 'double' 
        if (dueling_dqn):
            add_str = 'dueling'
        checkpoint_path = "models/%s_%s_%d.pth" %(str(env_id), add_str, checkpoint)
        Q.load_state_dict(torch.load(checkpoint_path))
        Q_target.load_state_dict(torch.load(checkpoint_path))
        print('load model success')

    # initialize optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # create replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ########

    ###########
    # RUN ENV #
    ###########

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10
    SAVE_MODEL_EVERY_N_STEPS = 100
    episode_rewards = []
    episode_reward = 0

    for t in itertools.count(start=checkpoint):

        ### Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### Step the env and store the transition
        # store last frame, return idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # get observations to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()
        #print(observations.shape)

        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs = torch.from_numpy(observations).unsqueeze(0).type(dtype) / 255.0
                #print(obs.shape)
                q_value_all_actions = Q(Variable(obs, volatile=True)).cpu()
                action = ((q_value_all_actions).data.max(1)[1])[0]
            else:
                action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]
            
        obs, reward, done, stop, emergence_break = env.step(action)
        episode_reward += reward

        # store effect of action 
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        # reset env if reached episode boundary
        if done:
            obs = env.reset()
            episode_rewards.append(episode_reward)
            print("current episode reward %d" % episode_reward)
            episode_reward = 0

        # update last_obs
        last_obs = obs

        ### Perform experience replay and train the network
        # if the replay buffer contains enough samples..
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(batch_size)
            obs_t = Variable(torch.from_numpy(obs_t)).type(dtype) / 255.0
            act_t = Variable(torch.from_numpy(act_t)).type(dlongtype)
            rew_t = Variable(torch.from_numpy(rew_t)).type(dtype)
            obs_tp1 = Variable(torch.from_numpy(obs_tp1)).type(dtype) / 255.0
            done_mask = Variable(torch.from_numpy(done_mask)).type(dtype)

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            q_values = Q(obs_t)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            if (double_dqn):

                #------------
                # double DQN
                #------------

                # get Q values for best actions in obs_tp1
                # based off the current Q network
                # max(Q(s',a',theta_i)) wrt a'
                q_tp1_values = Q(obs_tp1).detach()
                _, a_prime = q_tp1_values.max(1)

                # get Q values from frozen network for next state and chosen action
                # Q(s', argmax(Q(s',a',theta_i), theta_i_frozen)) (argmax wrt a')
                q_target_tp1_values = Q_target(obs_tp1).detach()
                q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()

                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime

                error = rew_t + gamma * q_target_s_a_prime - q_s_a
            
            else:
                #-------------
                # regular DQN
                #-------------

                # get Q values for best actions in obs_tp1
                # based off frozen Q network
                # max(Q(s',a',theta_i_frozen)) wrt a'
                q_tp1_values = Q_target(obs_tp1).detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)

                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done_mask) * q_s_a_prime

                # Compute Bellman error
                # r + gamma * Q(s', a', theta_i_frozen) - Q(s, a, theta_i)
                error = rew_t + gamma * q_s_a_prime - q_s_a

            # clip the error and flip
            clipped_error = -1.0 * error.clamp(-1, 1)

            # backwards pass
            optimizer.zero_grad()
            #q_s_a.backward(clipped_error.data.unsqueeze(1))
            q_s_a.backward(clipped_error.data)

            # update
            optimizer.step()
            num_param_updates += 1

            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            # (2) Log values and gradients of the parameters (histogram)
            #if t % LOG_EVERY_N_STEPS == 0:
                #for tag, value in Q.named_parameters():
                   # tag = tag.replace('.', '/')
                    #logger.histo_summary(tag, to_np(value), t+1)
                    #logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)
            #####

        ### 4. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            add_str = ''
            if (double_dqn):
                add_str = 'double' 
            if (dueling_dqn):
                add_str = 'dueling'
            model_save_path = "models/%s_%s_%d.pth" %(str(env_id), add_str, t)
            torch.save(Q.state_dict(), model_save_path)

        #episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-10:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward (10 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush()







