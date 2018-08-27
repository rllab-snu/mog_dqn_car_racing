import distdeepq
import gym
import numpy as np
from baselines import logger
from baselines.common.atari_wrappers import wrapper_car_racing

def exp(env_name='CarRacing-v0',
        lr=1e-4,
        max_timesteps=25e6,
        buffer_size=1e6,
        batch_size=32,
        exp_t1=1e6,
        exp_p1=0.1,
        exp_t2=25e6,
        exp_p2=0.01,
        train_freq=4,
        learning_starts=5e4,
        target_network_update_freq=1e4,
        gamma=0.99,
        num_cpu=50,
        nb_atoms=5,
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        action_res=None
        ):

    env = gym.make(env_name)
    env = wrapper_car_racing(env)  # frame stack
    # logger.configure(dir=os.path.join('.', datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")))
    logger.configure()

    n_action, action_map = get_action_information(env, env_name, action_res=action_res)

    model = distdeepq.models.cnn_to_dist_mlp(
        convs=convs, # [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=hiddens, # [512],
        # n_action=n_action,
        dueling=False
    )
    act = distdeepq.learn(
        env,
        p_dist_func=model,
        lr=lr,  # 1e-4
        max_timesteps=int(max_timesteps), # 25M
        buffer_size=int(buffer_size), # 1M
        batch_size=int(batch_size),
        exp_t1=exp_t1,
        exp_p1=exp_p1,
        exp_t2=exp_t2,
        exp_p2=exp_p2,
        train_freq=train_freq,
        learning_starts=learning_starts, # 50000
        target_network_update_freq=target_network_update_freq, # 10000
        gamma=gamma,
        num_cpu=num_cpu,
        prioritized_replay=False,
        dist_params={'nb_atoms': nb_atoms},
        n_action=int(n_action),
        action_map=action_map
    )
    act.save("car_racing_model.pkl")
    env.close()


def get_action_information(env, env_name, action_res=None):
    action_map = []
    if isinstance(env.action_space, gym.spaces.Box):
        if env_name == "CarRacing-v0":
            action_map = np.zeros([np.prod(action_res), 3])
            ste = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])  # -1~1
            gas = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])  # 0~1
            brk = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])  # 0~1
            for i in range(action_res[0]):
                for j in range(action_res[1]):
                    for k in range(action_res[2]):
                        s = action_res[2] * action_res[1] * i + action_res[2] * j + k
                        action_map[s, :] = [ste[i], gas[j], brk[k]]
        n_action = np.prod(action_res)

    else:
        raise NotImplementedError("action space not supported")

    return n_action, action_map


if __name__ == '__main__':
    exp(buffer_size=1e5, action_res=[5, 5, 5])
