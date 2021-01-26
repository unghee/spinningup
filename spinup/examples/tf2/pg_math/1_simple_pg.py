import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

import numpy as np
import gym
from gym.spaces import Discrete, Box

def mlp(sizes, activation='relu', output_activation='linear'):
    # Build a feedforward neural network.
    model = tf.keras.Sequential()

    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        model.add(tf.keras.layers.Dense(sizes[j+1], activation=act))
    return model

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs, training=True)
        # logits = tf.squeeze(logits, axis=0)
        return tfp.distributions.Categorical(logits=logits,dtype=tf.float32)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return int(get_policy(obs).sample().numpy())

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -tf.reduce_mean(logp * weights)
    def reward_to_go(rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    # make optimizer
    optimizer = Adam(learning_rate=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            # obs=np.reshape(obs,(-1,np.shape(obs)[0]))
            # obs= np.reshape(obs,(-1, obs_dim))
            batch_obs.append(obs.copy())
   
            # act in the environment
            act = get_action(np.reshape(obs,(-1, obs_dim)))
            # act = get_action(tf.convert_to_tensor(obs, dtype=tf.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len
                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                # batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        with tf.GradientTape() as tape:
            batch_loss = compute_loss(obs=tf.convert_to_tensor(batch_obs, dtype=tf.float32),
                act=tf.convert_to_tensor(batch_acts, dtype=tf.int32),
                weights=tf.convert_to_tensor(batch_weights, dtype=tf.float32)
                )
        # var_list_fn = lambda: model.trainable_weights
        optimizer.minimize(loss=batch_loss,var_list=logits_net.trainable_variables,tape=tape)
        # gradients = tape.gradient(batch_loss, logits_net.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, logits_net.trainable_variables))

        # logits_net.compile(loss=batch_loss,optimizer=optimizer)

        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)