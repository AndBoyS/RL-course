import time
import numpy as np
from joblib import Parallel, delayed
from IPython.display import clear_output
from matplotlib import pyplot as plt
import gymnasium as gym


def get_reseted_env(env_name):
    return gym.make(env_name, render_mode="rgb_array").env


def generate_session(agent, env_name, t_max=1000, test=False):
    """
    Play a single game using agent neural network.
    Terminate when game finishes or after :t_max: steps
    """
    states, actions = [], []
    total_reward = 0
    env = get_reseted_env(env_name)
    n_actions = env.action_space.n
    possible_actions = range(n_actions)

    s, _ = env.reset()

    for t in range(t_max):

        # use agent to predict a vector of action probabilities for state :s:
        probs = agent.predict_proba(s.reshape(1, -1)).reshape(-1)

        assert probs.shape == (n_actions,), f"make sure probabilities are a vector (hint: np.reshape), got {probs.shape}"

        # use the probabilities you predicted to pick an action
        if test:
            # on the test use the best (the most likely) actions at test
            # experiment, will it work on the train and vice versa?
            a = np.argmax(probs)
        else:
            # sample proportionally to the probabilities,
            # don't just take the most likely action at train
            a = np.random.choice(possible_actions, p=probs)

        new_s, r, done, info, _ = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you are confused, see examples below. Please don't assume that states are integers
    (they will become different later).
    """

    reward_threshold = np.percentile(rewards_batch, q=percentile)
    elite_states, elite_actions = [], []
    for reward, states, actions in zip(rewards_batch, states_batch, actions_batch):
        if reward > reward_threshold:
            elite_states.extend([np.squeeze(state) for state in states])
            elite_actions.extend(actions)

    # Let's see that it differs from tabular `select_elites`.
    # Our neural network now is trained with one object and it takes an input of `(1, n_states)`.
    # So, we should use `np.squeeze` to remove leading dimension in the deep `select_elites`.

    return elite_states, elite_actions


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


def parallel_func_wrapper(func, *args, **kwargs):
    return Parallel(n_jobs=3, prefer="threads")(delayed(func)(*args, **kwargs))


def train(agent,
          env_name,
          epochs,
          n_sessions_in_epoch=100,
          percentile=70,
          win_value=None,
          reward_min=None,
          parallel=False,
          n_jobs=4,
          ):

    log = []

    for _ in range(epochs):
        # generate new sessions

        t1 = time.time_ns()
        if parallel:
            sessions = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(generate_session)(agent, env_name) for _ in range(n_sessions_in_epoch)
            )
        else:
            sessions = [generate_session(agent, env_name) for _ in range(n_sessions_in_epoch)]

        states_batch, actions_batch, rewards_batch = map(lambda x: np.array(x, dtype=object),
                                                         zip(*sessions))
        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch)

        agent.partial_fit(elite_states, elite_actions)

        t2 = time.time_ns()

        if reward_min is None:
            reward_min = np.min(rewards_batch)
        show_progress(rewards_batch, log, percentile, reward_range=[reward_min, np.max(rewards_batch)])
        print(f'Training time for last epoch: {(t2 - t1)/10**9:.2f} seconds')
        print()

        if win_value is not None and np.mean(rewards_batch) > win_value:
            print("You Win!")
            return