import argparse


def get_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
    parser.add_argument("--delta_steps", type=int, default=int(1e3))
    # parser.add_argument("--delta_steps", type=int, default=100)

    # parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
    parser.add_argument("--buffer_capacity", type=int, default=int(1e4), help="The maximum replay-buffer capacity ")
    # parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    
    parser.add_argument("--hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5), help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="Important sampling parameter in PER")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=False, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=False, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=False, help="Whether to use n_steps Q-learning")


    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pre_policy", type=str, default="checkpoints_single_v3_easy_data/epoch_5.pt")
    
    parser.add_argument("--root", type=str, default="/home/zhaishichao/vlfm")
    parser.add_argument("--model_file_name", type=str, default="Models_train_PPO")
    parser.add_argument("--experiment_details", type=str, default="new_replay_buffer_multi_process_q_learning")



    # parse arguments
    args = parser.parse_args()
    return args

args = get_args()