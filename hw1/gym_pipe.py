import time
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from joblib import Parallel, delayed
from IPython.display import clear_output
from matplotlib import pyplot as plt
import gymnasium as gym


class GymPipeline:

    # Сколько нужно среднего реварда для победы
    win_value_dict = {
        'CartPole-v1': 190,
        'LunarLander-v2': 50,
    }

    def __init__(
            self,
            env_name,
            agent=None,
        ):
        self.env_name = env_name

        env = self.get_env()
        # initialize agent to the dimension of state space and number of actions
        self.n_actions = env.action_space.n
        self.possible_actions = list(range(self.n_actions))
        self.agent = agent

        self.win_value = self.win_value_dict[env_name]

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, agent):
        if agent is not None:
            env = self.get_env()
            s = env.reset()[0]
            agent = deepcopy(agent)
            agent.partial_fit([s] * self.n_actions, self.possible_actions, self.possible_actions)

        self._agent = agent
        self._agent_base = deepcopy(agent)

    def reset_agent(self):
        self._agent = deepcopy(self._agent_base)

    def get_env(self):
        return gym.make(self.env_name, render_mode="rgb_array").env

    def generate_session(self, max_iter_per_session, test=False):
        """
        Play a single game using agent neural network.
        Terminate when game finishes or after :t_max: steps
        """
        states, actions = [], []
        total_reward = 0

        env = self.get_env()
        agent = deepcopy(self.agent)
        s, _ = env.reset()

        for t in range(max_iter_per_session):

            # use agent to predict a vector of action probabilities for state :s:
            probs = agent.predict_proba(s.reshape(1, -1)).reshape(-1)

            assert probs.shape == (self.n_actions,), f"make sure probabilities are a vector (hint: np.reshape), got {probs.shape}"

            # use the probabilities you predicted to pick an action
            if test:
                # on the test use the best (the most likely) actions at test
                # experiment, will it work on the train and vice versa?
                a = np.argmax(probs)
            else:
                # sample proportionally to the probabilities,
                # don't just take the most likely action at train
                a = np.random.choice(self.possible_actions, p=probs)

            new_s, r, done, info, _ = env.step(a)

            # record sessions like you did before
            states.append(s)
            actions.append(a)
            total_reward += r

            s = new_s
            if done:
                break

        env.close()
        return states, actions, total_reward

    def select_elites(self, states_batch, actions_batch, rewards_batch, session_percentile):
        """
        Select states and actions from games that have rewards >= session_percentile
        :param states_batch: list of lists of states, states_batch[session_i][t]
        :param actions_batch: list of lists of actions, actions_batch[session_i][t]
        :param rewards_batch: list of rewards, rewards_batch[session_i]

        :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

        Please return elite states and actions in their original order
        [i.e. sorted by session number and timestep within session]

        If you are confused, see examples below. Please don't assume that states are integers
        (they will become different later).
        """

        reward_threshold = np.percentile(rewards_batch, q=session_percentile)
        elite_states, elite_actions = [], []
        for reward, states, actions in zip(rewards_batch, states_batch, actions_batch):
            if reward > reward_threshold:
                elite_states.extend([np.squeeze(state) for state in states])
                elite_actions.extend(actions)

        # Let's see that it differs from tabular `select_elites`.
        # Our neural network now is trained with one object and it takes an input of `(1, n_states)`.
        # So, we should use `np.squeeze` to remove leading dimension in the deep `select_elites`.
        return elite_states, elite_actions

    def train(self,
              epochs=150,
              n_sessions_in_epoch=100,
              max_iter_per_session=1000,
              percentile=70,
              n_jobs=1,
              seed=None,
              ):

        assert self.agent is not None, 'Agent attr needs to be set for training'
        self.reset_agent()

        log = []

        for _ in range(epochs):

            t1 = time.time_ns()
            # generate new sessions
            sessions = self.generate_sessions(n_sessions_in_epoch, max_iter_per_session, n_jobs)

            states_batch, actions_batch, rewards_batch = map(lambda x: np.array(x, dtype=object),
                                                             zip(*sessions))

            elite_states, elite_actions = self.select_elites(states_batch, actions_batch, rewards_batch, percentile)
            self.agent.partial_fit(elite_states, elite_actions)

            t2 = time.time_ns()

            reward_range = rewards_batch.min(), rewards_batch.max()
            show_progress(rewards_batch, log, percentile, reward_range=reward_range)
            epoch_time = (t2 - t1) / 10**9
            print(f'Training time for last epoch: {epoch_time:.2f} seconds')
            print()

            if self.win_value is not None and np.mean(rewards_batch) > self.win_value:
                print("You Win!")
                return

    def generate_sessions(self, n_sessions_in_epoch, max_iter_per_session, n_jobs):
        if n_jobs > 1:
            with Pool(n_jobs) as p:
                sessions = p.map(self.generate_session, [max_iter_per_session] * n_sessions_in_epoch)
        else:
            sessions = [self.generate_session(max_iter_per_session) for _ in range(n_sessions_in_epoch)]

        return sessions


class GymPipelineOptimized(GymPipeline):
    # Будем заново использовать сессии с топ наградами на эпохах
    prev_sessions_to_reuse = None

    def __init__(
            self,
            env_name,
            agent=None,
            top_rewards_to_reuse=20,
    ):
        super().__init__(env_name, agent)
        self.top_rewards_to_reuse = top_rewards_to_reuse

    def generate_sessions(self, n_sessions_in_epoch, max_iter_per_session, n_jobs):
        # Считаем на top_rewards_to_reuse меньше сессий, используя сессии с топ наградами с прошлой эпохи

        is_first_iter = self.prev_sessions_to_reuse is None

        if not is_first_iter:
            max_iter_per_session -= self.top_rewards_to_reuse

        sessions = super().generate_sessions(n_sessions_in_epoch, max_iter_per_session, n_jobs)
        top_idx = np.argsort([reward for _, _, reward in sessions])[self.top_rewards_to_reuse:]

        sessions_to_reuse = select_idx(sessions, top_idx)

        if not is_first_iter:
            sessions.extend(self.prev_sessions_to_reuse)

        self.prev_sessions_to_reuse = sessions_to_reuse

        return sessions


def select_idx(l, idx):
    return [l[i] for i in idx]


def show_progress(rewards_batch, log, percentile, reward_range=(-990, +10)):
    """
    A convenience function that displays training progress.
    No cool math here, just charts.
    """

    mean_reward = np.mean(rewards_batch)
    threshold = np.percentile(rewards_batch, percentile)
    log.append([mean_reward, threshold])

    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label='Mean rewards')
    plt.plot(list(zip(*log))[1], label='Reward thresholds')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(rewards_batch, range=reward_range)
    plt.vlines([np.percentile(rewards_batch, percentile)],
               [0], [100], label="percentile", color='red')
    plt.legend()
    plt.grid()

    clear_output(True)
    plt.show()
    print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))

