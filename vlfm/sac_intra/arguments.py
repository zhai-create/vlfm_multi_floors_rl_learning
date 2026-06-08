import argparse


def get_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for SAC")
    # parser.add_argument("--delta_steps", type=int, default=int(1e3))
    # parser.add_argument("--delta_steps", type=int, default=int(200))
    # parser.add_argument("--delta_steps", type=int, default=int(40))
    parser.add_argument("--delta_steps", type=int, default=int(80))
    # parser.add_argument("--delta_steps", type=int, default=int(800))
    # parser.add_argument('--random_steps', type=int, default=5000, help='steps for random policy to explore')
    parser.add_argument('--random_steps', type=int, default=0, help='steps for random policy to explore')
    # parser.add_argument('--random_steps', type=int, default=1000, help='steps for random policy to explore')
    parser.add_argument("--real_batch_size", type=int, default=8)


    # parser.add_argument("--buffer_capacity", type=int, default=int(1e4), help="The maximum replay-buffer capacity ")
    parser.add_argument("--buffer_capacity", type=int, default=int(4e4), help="The maximum replay-buffer capacity ")
    # parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    # parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    # parser.add_argument("--batch_size", type=int, default=4096, help="batch size")
    # parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    # parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")

    parser.add_argument("--hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    # parser.add_argument("--actor_lr", type=float, default=3e-5, help="Learning rate of actor")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument('--adaptive_alpha', type=bool, default=True, help='Use adaptive alpha turning')
    # parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
    # parser.add_argument('--alpha', type=float, default=0.036, help='init alpha')
    # parser.add_argument('--alpha', type=float, default=0.001, help='init alpha')
    parser.add_argument('--alpha', type=float, default=0.01, help='init alpha')

    parser.add_argument('--curr_t_init', type=float, default=10)
    parser.add_argument('--curr_t_min', type=float, default=1)
    parser.add_argument('--curr_t_decay', type=float, default=0.00009)

    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    # parser.add_argument("--tau", type=float, default=0.001, help="soft update the target network")
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--pre_policy", type=str, default="checkpoints_single_v3_easy_data_softmax_resize/epoch_5.pt")
    # parser.add_argument("--pre_policy", type=str, default="checkpoints_gt_perception_value_and_obstacle_resize/epoch_10.pt")
    # parser.add_argument("--pre_policy", type=str, default="checkpoints_gt_perception_multi_floors_il_frozen/epoch_18.pt")
    parser.add_argument("--pre_policy", type=str, default="checkpoints_gt_perception_intra_il/epoch_10.pt")
    
    parser.add_argument("--root", type=str, default="/home/zhaishichao/Data/vlfm_multi_floors_rl_learning")
    parser.add_argument("--model_file_name", type=str, default="Models_train_PPO_intra")
    parser.add_argument("--experiment_details", type=str, default="multi_process_sac")
    parser.add_argument("--critic_experiment_details", type=str, default="multi_process_critic")

    parser.add_argument("--buffer_data_folder", type=str, default="sac_buffer_data_multi_process")
    

    # parse arguments
    args = parser.parse_args()
    return args

args = get_args()