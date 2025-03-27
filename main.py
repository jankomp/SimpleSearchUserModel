from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from gym_examples.envs.simple_search import SimpleSearchEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import numpy as np
import os

class CustomMLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomMLPPolicy, self).__init__(*args, **kwargs,
                                              net_arch=dict(pi=[256], vf=[256]),
                                              activation_fn=th.nn.ReLU)
        
def make_env(render_mode = None, search_tree_depth=4, patience_penalty=1.0, cognitive_slowness=3):
    def _init():
        return SimpleSearchEnv(render_mode=render_mode, search_tree_depth=search_tree_depth, patience_penalty=patience_penalty, cognitive_slowness=cognitive_slowness)
    return _init

def test_configurations():
    if not os.path.exists("models"):
        os.makedirs("models")

    patience_penalties = [0.0, 0.1, 0.5, 1.0, 2.0]
    cognitive_slownesses = [0, 1, 5, 10]
    search_tree_depths = [4, 8]
    results = {}

    for search_tree_depth in search_tree_depths:
        for patience_penalty in patience_penalties:
            for cognitive_slowness in cognitive_slownesses:
                print(f"Testing configuration: patience_penalty={patience_penalty}, cognitive_slowness={cognitive_slowness}, tree_depth={search_tree_depth}")
                env = make_vec_env(make_env(search_tree_depth=search_tree_depth, patience_penalty=patience_penalty, cognitive_slowness=cognitive_slowness) , n_envs=8, vec_env_cls=SubprocVecEnv)
                model = PPO(CustomMLPPolicy, env, verbose=0, tensorboard_log="./ppo_simplesearch_tensorboard/")
                model.learn(total_timesteps=1_000_000, tb_log_name=f"patience_{patience_penalty}_cognitive_{cognitive_slowness}_depth_{search_tree_depth}")
                
                model_name = f"ppo_simple_search_pcd_{patience_penalty}_{cognitive_slowness}_{search_tree_depth}"
                model.save(os.path.join("models", model_name))

                del model
                model = PPO.load(os.path.join("models", model_name))

                eval_env = make_vec_env(make_env(search_tree_depth=search_tree_depth, patience_penalty=patience_penalty, cognitive_slowness=cognitive_slowness), n_envs=1, vec_env_cls=DummyVecEnv)
                obs = eval_env.reset()
                
                action_distribution = np.zeros(eval_env.action_space.n)
                total_episode_lengths = []
                total_final_node_values = []

                for _ in range(100):  # Evaluate for 100 episodes
                    obs = eval_env.reset()
                    episode_length = 0
                    done = False

                    true_value = 0
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        #print(f"{eval_env.env_method("log", indices=0)} Action: {np.squeeze(action)}")
                        action_distribution[action] += 1
                        obs, rewards, dones, info  = eval_env.step(action)
                        done = dones[0]
                        episode_length += 1
                        if not done:
                            true_value = np.squeeze(eval_env.env_method("get_current_node_value", indices=0))

                    total_final_node_values.append(true_value)
                    total_episode_lengths.append(episode_length)
                
                # Save metrics for this configuration
                results[(patience_penalty, cognitive_slowness, search_tree_depth)] = {
                    "action_distribution": action_distribution / np.sum(action_distribution),  # Normalize
                    "average_episode_length": np.mean(total_episode_lengths),
                    "average_final_node_value": np.mean(total_final_node_values),
                }
                print(f"Results for configuration: patience_penalty={patience_penalty}, cognitive_slowness={cognitive_slowness}, search_tree_depth={search_tree_depth}: {results[(patience_penalty, cognitive_slowness, search_tree_depth)]}")

    print("Evaluation Results:")
    for config, metrics in results.items():
        print(f"Config {config}: {metrics}")

    # save results in a csv file
    with open("results.csv", "w") as f:
        f.write("patience_penalty,cognitive_slowness,search_tree_depth,action_home,action_parent,action_left,action_right,action_stay,action_end,average_episode_length,average_final_node_value\n")
        for config, metrics in results.items():
            f.write(f"{config[0]},{config[1]},{config[2]},{metrics['action_distribution'][0]},{metrics['action_distribution'][1]},{metrics['action_distribution'][2]},{metrics['action_distribution'][3]},{metrics['action_distribution'][4]},{metrics['action_distribution'][5]},{metrics['average_episode_length']},{metrics['average_final_node_value']}\n")

def train_and_test():
    env = make_vec_env(make_env(), n_envs=8, vec_env_cls=SubprocVecEnv)
    model = PPO(CustomMLPPolicy, env, verbose=1, tensorboard_log="./ppo_simplesearch_tensorboard/")
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_simple_search")
    del model
    model = PPO.load("ppo_simple_search")

    eval_env = make_vec_env(make_env(), n_envs=1, vec_env_cls=DummyVecEnv)
    obs = eval_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        #print(f"{eval_env.env_method("log", indices=0)} Action: {np.squeeze(action)}")
        obs, rewards, dones, info = eval_env.step(action)
        if dones.all():
            obs = eval_env.reset()
    env.close()
    

if __name__ == "__main__":
    test_configurations()