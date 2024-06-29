


import argparse
from itertools import count
import logging
import math
import os
import time
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from config import *
from user_model.preference_model import EvolvingPreferenceModeling
from utils import *
from env.user_simulator import UserSimulator
from env.system_env import RecommendationEnv
from rl_ppo.ppo_crs import Agent
from rl_ppo.replay_buffer import ReplayBuffer
from rl_ppo.ppo_eval import ppo_evaluate



def train(args, retrieval_graph, filename):
    set_random_seed(args.seed)
    
    user_count, item_count, attribute_count = retrieval_graph.node_counts['user'], retrieval_graph.node_counts['item'], retrieval_graph.node_counts['attribute']
    user_id_max, item_id_max, attribute_id_max = max(retrieval_graph.node_ids_by_type['user']), max(retrieval_graph.node_ids_by_type['item']), max(retrieval_graph.node_ids_by_type['attribute'])
    print('user_count: {}, item_count: {}, attribute_count: {}'.format(user_count, item_count, attribute_count))
    print('user_id_max: {}, item_id_max: {}, attribute_id_max: {}'.format(user_id_max, item_id_max, attribute_id_max))
    

    # System Rnking Model
    system_model = EvolvingPreferenceModeling(user_id_max+1, item_id_max+1, attribute_id_max+1, args.hidden_size, args.seq_model, device=args.device)
    
    
    system_model = load_EPM_model(args.data_name, system_model, args.ranking_model_mv)
    system_model.to(args.device)

    #Load training embedding
    user_embeds = system_model.user_embeds.weight.data[:user_count,:]
    item_embeds = system_model.item_embeds.weight.data[:item_count,:]
    attribute_embeds = system_model.attribute_embeds.weight.data[:attribute_count,:]
    print(f"user_embeds shape: {user_embeds.shape}, item_embeds shape: {item_embeds.shape}, attribute_embeds shape: {attribute_embeds.shape}")


 
    args.state_size = system_model.hidden_size

    # Initialize rl agent
    
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)
    agent = Agent(args, replay_buffer)

    # Initialize & Load RL model
    agent_AvgT_best = args.max_turn  # only save the best model
    start_time = time.time()
    if args.load_rl_epoch != 0 :
        agent.load_rl_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        final_formatted_metrics, final_formatted_SR_turn_15, test_user_num = ppo_evaluate(args, agent, retrieval_graph, item_embeds, attribute_embeds, system_model)

        logging.basicConfig(filename=f'{args.mv}[{args.data_name}]-load_rl_epoch_{args.load_rl_epoch}.log', format='%(asctime)s - %(name)s - %(message)s',
                            datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO)
        logging.info(f"================pid: {os.getpid()}================")
        # metrics = [SR5, SR10, SR15, AvgT, Rank, total_reward]
        format_logging(final_formatted_metrics, logging, total_steps, mode='test')
        save_turn_metric(args.data_name, filename, total_steps, test_user_num, final_formatted_SR_turn_15)
        save_rl_metric(args.data_name, filename, total_steps, final_formatted_metrics,
                       spend_time=time.time() - start_time, mode='test')
        return
    else:
        if os.path.exists(f'{filename}.log'):
            os.remove(f'{filename}.log')

    # === write logging ==
    logging.basicConfig(filename=f'{args.mv}[{args.data_name}].log', format = '%(asctime)s - %(name)s - %(message)s', datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO)
    logging.info(f"================pid: {os.getpid()}================")
    args_dict = vars(args)
     
    logging.info('===== %s Namespace =====', args.data_name)
    log_line = ' '.join(['%s=%s | ' % (k, v) for k, v in args_dict.items()])
    logging.info(log_line)
    logging.info('=============================')
    
    # 打印logging信息
    print("logging info",logging.getLogger().handlers)
    writer = SummaryWriter(log_dir=f'./tensorboard/{filename}')
    
    # Initialize environment
    env =  RecommendationEnv(args, retrieval_graph, mode='train', ranking_model=system_model)
    ui_train_data = load_ui_data(args.data_name, 'valid')
    user_ids = list(ui_train_data.keys())

    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training
    buffer_new_count = 0  # Record the number of transitions in buffer

    def padding_embedding(action_embeds: torch.Tensor, action_ind: list, dim_size: int) -> torch.Tensor:
        assert isinstance(action_embeds, torch.Tensor), "action_embeds must be a torch.Tensor"
        assert isinstance(action_ind, list), "action_ind must be a list"
        assert isinstance(dim_size, int) and dim_size > 0, "dim_size must be a positive integer"

        current_size = action_embeds.shape[0]
        if current_size < dim_size:
            padding_size = dim_size - current_size
            new_embeds = F.pad(action_embeds, (0, 0, 0, padding_size), mode='constant', value=0)
            new_ind = action_ind + [-1] * padding_size
            return new_embeds, new_ind

        return action_embeds, action_ind

    with trange(args.max_train_steps) as tqrange:
        SR5, SR10, SR15, AvgT, Rank, total_reward = [0.] * 6
        avg_user_prefer_precision = 0.
        while total_steps < args.max_train_steps:
            if args.block_conversation:
                blockPrint() # TODO Disable print function
            # Reset the user simulator tot start a new conversation
            # print('\n================ Start {}-th conversation ================'.format(total_steps))
            user_id = np.random.choice(user_ids).astype(int)
            target_items = np.random.choice(ui_train_data[str(user_id)], args.target_item_num, replace=False).astype(int).tolist()
            user_simulator = UserSimulator((user_id, target_items),args, retrieval_graph, mode='train')
            

            user_id, explicit_attributes_ids, item_topk_ids, attribute_topk_ids = env.reset(user_simulator)
            
            # ================== Printing  Conversation==================
            if args.block_conversation is False:
                env.render_init()  # TODO conversation printing
            # ==========================================================
            state_emb = env.get_state_embedding() # (1, hidden_size)
          
            epi_reward = 0
            turn_user_precision , ask_turn_num = 0., 0
            for t in count(start=1):   # user  dialog
                # choose action
            
 
                cand_item_embs,  item_topk_ids= padding_embedding(item_embeds[item_topk_ids], item_topk_ids, args.topk_item) # (k1, D)
                cand_attribute_embs, attribute_topk_ids = padding_embedding(attribute_embeds[attribute_topk_ids], attribute_topk_ids, args.topk_attribute) # (k2, D)
                
   
                cand_embs = torch.cat((cand_item_embs, cand_attribute_embs), dim=0).unsqueeze(dim=0) # (1, k1+k2,D)
                decision_ind, selected_actions, final_logprob, _ = agent.choose_action(state_emb, cand_embs, item_topk_ids+attribute_topk_ids, train=True, topk_item=args.topk_item, topk_attr=args.topk_attribute, rec_num=args.rec_num, ask_num=args.ask_num)


                # execute action
                user_id, next_explicit_attributes_ids, next_item_topk_ids, next_attribute_topk_ids, reward, done, rec_accept_inds = env.world_model_step(decision_ind, selected_actions)
                next_state_emb = env.get_state_embedding()

                # ================== Printing  Conversation==================
                if args.block_conversation is False:
                    env.render()  # TODO conversation printing
                    pass
                # ==========================================================
                epi_reward += reward

           

                """
                s:  "user_id", "attribute_sequence_ids", 
                s_:  "user_id",  'next_attribute_sequence_ids',
                a:  "topk_item_ids", "topk_attributes_ids" ; 'decision_ind', 'action_inds' 
                """
                # ["user_id", "attribute_sequence_ids", 'next_attribute_sequence_ids', 'decision_ind', 'action_inds', 'a_logprob', "action_embs", "reward", "done", "rec_done"]
                agent.replay_buffer.add(
                    user_id,   # (1, )
                    state_emb.squeeze(dim=0),   # (D+T+T+20, )
                    next_state_emb.squeeze(dim=0),  # (D+T+T+20, )
                    final_logprob,  # (1, )
                    cand_embs.squeeze(dim=0), #(k1+k2, D)
                    reward, # (1, )
                    done, # (1, )
                    True if rec_accept_inds else False # (1, )
                )
                


                # update state
                state_emb = next_state_emb
                item_topk_ids = next_item_topk_ids
                attribute_topk_ids = next_attribute_topk_ids

                buffer_new_count += 1
                

                # When the number of transitions in buffer reaches buffer_size,then update
                if buffer_new_count == args.buffer_size:

                    # enablePrint() # Enable print function
                    actor_loss, critic_loss, entropy = agent.update(total_steps)
                    writer.add_scalar("losses/actor_loss", actor_loss, total_steps)
                    writer.add_scalar("losses/critic_loss", critic_loss, total_steps)
                    writer.add_scalar("losses/entropy", entropy, total_steps)
                    save_rl_loss_log(dataset=args.data_name, filename=filename, epoch=total_steps,
                                      epoch_loss=[actor_loss, critic_loss, entropy])
                    logging.info(f"Step: {total_steps} - Actor_loss: {actor_loss:.4f} - Critic_loss: {critic_loss:.4f} - Entropy: {entropy:.4f}")

                    buffer_new_count = 0
                if decision_ind == 1:
                    ask_turn_num += 1
                    turn_user_precision += env.compute_current_precision()

                if done:
                    # update metric
                    if rec_accept_inds:  # recommend successfully
                        if t < 5:
                            SR5 += 1
                            SR10 += 1
                            SR15 += 1
                        elif t < 10:
                            SR10 += 1
                            SR15 += 1
                        else:
                            SR15 += 1
                        Rank += ndcg_at_k(rec_accept_inds, args.rec_num)
                    else:
                        Rank += 0
                    AvgT += t
                    total_reward += epi_reward
                    break
            enablePrint() # Enable print function
            avg_user_prefer_precision += turn_user_precision / ask_turn_num if ask_turn_num else 0
    
            total_steps += 1
            # Observe training result:
            if total_steps % args.observe_num == 0:
                writer.add_scalar("Metric/train-Avg_Precision", avg_user_prefer_precision/args.observe_num, total_steps)
                logging.info(f"train Step: {total_steps} - Avg_Precision: {avg_user_prefer_precision/args.observe_num:.4f}")
                avg_user_prefer_precision = 0.
                

                formatted_metrics = format_metrics(SR5, SR10, SR15, AvgT, Rank, total_reward, args.observe_num)
                format_logging(formatted_metrics, logging, total_steps, mode='train')
                format_writer(formatted_metrics, writer, total_steps, mode='train')
                save_rl_metric(args.data_name, filename, total_steps, formatted_metrics,
                               spend_time=time.time() - start_time, mode='train')

                # Reset metric
                SR5, SR10, SR15, AvgT, Rank, total_reward = [0.] * 6

                if agent_AvgT_best > formatted_metrics[3]:
                    agent_AvgT_best = formatted_metrics[3]
                    agent.save_rl_model(data_name=args.data_name, filename=filename, epoch_user=-1)
                    print(f'Successfully saved the best model at step:{total_steps}!')

            # Evaluate
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                final_formatted_metrics, final_formatted_SR_turn_15, test_user_num, test_avg_user_prefer_precision = ppo_evaluate(args, agent, retrieval_graph, item_embeds, attribute_embeds, system_model)
                writer.add_scalar("Metric/test-Avg_Precision", test_avg_user_prefer_precision, total_steps)
                logging.info(f"Test-Step: {total_steps} - Avg_Precision: {test_avg_user_prefer_precision:.4f}")
                # metrics = [SR5, SR10, SR15, AvgT, Rank, total_reward]
                format_logging(final_formatted_metrics, logging, total_steps, mode='test')
                
                format_writer(final_formatted_metrics, writer, total_steps, mode='test')
                save_rl_metric(args.data_name, filename, total_steps, final_formatted_metrics,
                               spend_time=time.time() - start_time, mode='test')
                save_turn_metric(args.data_name, filename, total_steps, test_user_num, final_formatted_SR_turn_15)

                # Save the metirc
                if evaluate_num % args.save_freq == 0:
                    save_numpy_metric(args.data_name, filename, total_steps, final_formatted_metrics, final_formatted_SR_turn_15)
            tqrange.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mv', type=str, default='TPCRS',
                        help='model version')
    parser.add_argument('--ranking_model_mv', type=str, default='user_model_v1',
                        help='model version')
    parser.add_argument('--data_name', type=str, default=LAST_FM_STAR, choices=[LAST_FM_STAR, YELP_STAR, BOOK],
                        help='One of { LAST_FM_STAR, YELP_STAR, BOOK}.')
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='2', help='gpu device.')

    

    # User-Centric Preference -- User simulator parameters
    parser.add_argument('--target_item_num', type=int, default=1, help='The number of target items')  # single interest for U-I data | multi-interest for U-I_multi data
    parser.add_argument('--intent_num', type=int, default=1, help='The number of intent for each user to start conversation')

    parser.add_argument('--personalization_alpha', type=float, default=0.5, help='The initial sample ratio of user personalized attributes')
    parser.add_argument('--adaptive_change', type=float, default=0.1, help='The change of sample ratio of target item attributes')
    parser.add_argument('--target_sample_strategy', type=str, default='inverse_weighted', choices=['inverse_weighted', 'weighted', 'uniform'])
    parser.add_argument('--his_sample_strategy', type=str, default='weighted', choices=['weighted', 'uniform'])

    # Item-centric Trick -- filtering strategy
    parser.add_argument('--filter_strategy', type=bool, default=False, help='filter items and attributes based on click or nonclick attributes')
    
    
    # action pruning parameters for RL
    parser.add_argument('--topk_item', type=int, default=10, help='K1: The number of pruning candidate items  (including ground-truth)  for RL')
    parser.add_argument('--topk_attribute', type=int, default=10, help='K2:The number of pruning candidate attributes  (including ground-truth)  for RL')
    parser.add_argument('--test_topk_item', type=int, default=10, help='The number of pruning candidate items for RL')
    parser.add_argument('--test_topk_attribute', type=int, default=10, help='The number of pruning candidate attributes for RL')

    # system recommendation parameters
    parser.add_argument('--block_conversation', type=bool, default=True, help='block conversation')
    parser.add_argument('--block_eval_conversation', type=bool, default=False, help='block conversation in evaluation')
    parser.add_argument('--max_turn', type=int, default=15, help='Max conversation turn')

    #system action parameters
    parser.add_argument('--rec_num', type=int, default=10, help='L1:The number of recommended items in a turn')
    parser.add_argument('--ask_num', type=int, default=2, help='L2:The number of features asked in a turn')
    parser.add_argument('--test_rec_num', type=int, default=10, help='The number of recommended items in a turn')
    parser.add_argument('--test_ask_num', type=int, default=2, help='The number of features asked in a turn')


    # RL parameters
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='the epoch of loading RL model')

    parser.add_argument("--max_train_steps", type=int, default=int(2e4), help=" Maximum number of training conversations")
    parser.add_argument('--observe_num', type=int, default=200, help='The number of steps to print metric') 
    parser.add_argument("--evaluate_freq", type=float, default=1e3,
                        help="Evaluate the policy every 'evaluate_freq' steps") 
    parser.add_argument('--test_size', type=int, default=1000, help='The number of users to test in evaluation') 
    parser.add_argument("--save_freq", type=int, default=1, help="Save metric frequency")

    parser.add_argument("--buffer_size", type=int, default=2048, help="Batch size") 
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size") 
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--action_size', type=int, default=64, help='action embedding size')

    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate.')
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    

    # state model parameters
    parser.add_argument('--seq_model', type=str, default='transformer', choices=['rnn', 'transformer', 'gru' , 'none'],
                        help='sequential learning method')
    parser.add_argument('--state_size', type=int, default=64, help='RL state embedding size')
    


    # RL tricks
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="advantage normalization")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_tanh", type=float, default=True, help="tanh activation function")

    args = parser.parse_args()
    print(os.getcwd())
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    

    retrieval_graph = load_graph(args.data_name)
    retrieval_graph.print_graph_info()

    filename = f'mv-{args.mv}_data-{args.data_name}'
    print("filename: ", filename)
    # === write args logging ==
    if args.load_rl_epoch == 0:
        print('Staring writing  model parameters')
        write_args(args.data_name, args, filename=filename, file_dir='RL-log-merge', args_name='Args',
                   open_type='w')
    train(args, retrieval_graph, filename)

if __name__ == '__main__':
    main()