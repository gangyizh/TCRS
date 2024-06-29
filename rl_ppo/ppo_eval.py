from itertools import count
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
from config import *
from env.system_env import RecommendationEnv
from env.user_simulator import UserSimulator
def ppo_evaluate(args, agent, retrieval_graph, item_embeds, attribute_embeds, system_model ):
    set_random_seed(args.seed)
    # Initialize environment

    test_env =  RecommendationEnv(args, retrieval_graph, mode='test', ranking_model=system_model)
    ui_test_data = load_ui_data(args.data_name, 'test')
    ui_list = [[int(user_str), item_id] for user_str, items in ui_test_data.items() for item_id in items]
    np.random.shuffle(ui_list)
    
    user_size = len(ui_list)
    test_size = args.test_size

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

    # Reset metric
    SR5, SR10, SR15, AvgT, Rank, total_reward = [0.] * 6
    SR_turn_15 = [0] * args.max_turn
    avg_user_prefer_precision = 0.
    test_users = test_size if test_size else user_size
    print(f"The selected test_users_set: {test_users} from test_set: {user_size}")

    for user_num in tqdm(range(test_users)):  # user_size
        user_id, target_items = ui_list[user_num][0], ui_list[user_num][1]
        user_simulator = UserSimulator((user_id, target_items),args, retrieval_graph, mode='test')
        if args.block_eval_conversation:
            blockPrint() # TODO Disable print function   
        # print('\n================test tuple:{}===================='.format(user_num))
        user_id, explicit_attributes_ids, item_topk_ids, attribute_topk_ids = test_env.reset(user_simulator)
        # ================== Printing  Conversation==================
        if args.block_eval_conversation is False:
            test_env.render_init()  # TODO conversation printing
        # ==========================================================

        epi_reward = 0
        turn_user_precision , ask_turn_num = 0., 0
        for t in count(start=1):   # user  dialog
            
            state_emb = test_env.get_state_embedding()

            cand_item_embs,  item_topk_ids= padding_embedding(item_embeds[item_topk_ids], item_topk_ids, args.test_topk_item) # (k1, D)
            cand_attribute_embs, attribute_topk_ids = padding_embedding(attribute_embeds[attribute_topk_ids], attribute_topk_ids, args.test_topk_attribute) # (k2, D)

            cand_embs = torch.cat((cand_item_embs, cand_attribute_embs), dim=0).unsqueeze(dim=0) # (1, k1+k2,D)
            latent_item_space, latent_attribute_space = test_env._get_latent_space()
            
            if latent_item_space <= args.test_rec_num:  #When Init item space is less than rec_num, we directly recommend all items
                decision_ind, selected_actions = 0, item_topk_ids
            else:
                decision_ind, selected_actions, _, _ = agent.choose_action(state_emb, cand_embs, item_topk_ids+attribute_topk_ids, train=False, topk_item=args.topk_item, topk_attr=args.topk_attribute, rec_num=args.test_rec_num, ask_num=args.test_ask_num)

            # execute action
            user_id, next_explicit_attributes_ids, next_item_topk_ids, next_attribute_topk_ids, reward, done, rec_accept_inds = test_env.step(decision_ind, selected_actions)
            # ================== Printing  Conversation==================
            if args.block_eval_conversation is False:
                test_env.render()  # TODO conversation printing
            # ==========================================================
            epi_reward += reward                                                        

            # update state
            item_topk_ids = next_item_topk_ids
            attribute_topk_ids = next_attribute_topk_ids

            if decision_ind == 1:
                ask_turn_num += 1
                turn_user_precision += test_env.compute_current_precision()

            if done:
                # update metric
                if rec_accept_inds:  # recommend successfully
                    SR_turn_15[t:] = [x + 1 for x in SR_turn_15[t:]]  # record SR for each turn
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

        avg_user_prefer_precision += turn_user_precision / ask_turn_num if ask_turn_num else 0
    
    enablePrint()
    avg_user_prefer_precision = avg_user_prefer_precision / user_num
    def format_metrics(SR5, SR10, SR15, AvgT, Rank, total_reward, observe_num):
        metrics = [SR5, SR10, SR15, AvgT, Rank, total_reward]
        return [metric / observe_num for metric in metrics]

    # final result
    final_formatted_metrics = format_metrics(SR5, SR10, SR15, AvgT, Rank, total_reward, user_num)
    final_formatted_SR_turn_15 = [sr / user_num for sr in SR_turn_15]
    


    return final_formatted_metrics, final_formatted_SR_turn_15, test_users, avg_user_prefer_precision