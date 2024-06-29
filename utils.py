import json
import os
import pickle
import sys
import dill
import networkx as nx
from config import *
import numpy as np
import random
import torch

def load_EPM_model(data_name, model, version):
    save_path = TMP_DIR[data_name] + '/ranking_model/'
    model.load_state_dict(torch.load(save_path + f'{version}_model.pt'))
    print('ranking_model loaded from {}'.format(save_path + f'{version}_model.pt'))
    return model

def load_simulaotr_dict_data(data_name, mode='train'):
    relation_dict_path = os.path.join(DATA_DIR[data_name], "relation_dict")
    with open(os.path.join(relation_dict_path, f"user-item_dict_{mode}.json"), 'r') as f:
        user_item_dict = json.load(f)
    with open(os.path.join(relation_dict_path, f"item-attribute_dict.json"), 'r') as f:
        item_attribute_dict = json.load(f)
    return user_item_dict, item_attribute_dict

def load_ui_data(dataset, mode='train'):
    ui_data_file = os.path.join(DATA_DIR[dataset], "UI_Interaction_data", f"review_dict_{mode}.json")
    with open(ui_data_file, 'r') as f:
        ui_data = json.load(f)
    return ui_data

def save_graph(graph, data_name):
    save_directory = TMP_DIR[data_name]
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_path = os.path.join(save_directory, 'retrieval_graph.pkl')
    with open(save_path, 'wb') as f:
        dill.dump(graph, f)
    print("Saving retrieval graph to", save_directory)

def save_simulator_user_attribute_dict(data_name, user_attribute_dict, mode='train'):
    save_directory = TMP_DIR[data_name] + '/user_simulator_user_attribute_dict/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_path = os.path.join(save_directory, f'user_attribute_dict_{mode}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(user_attribute_dict, f)
    print("Saving user attribute dict to", save_directory)

def load_simulator_user_attribute_dict(data_name, mode='train'):
    load_directory = TMP_DIR[data_name] + '/user_simulator_user_attribute_dict/'
    load_path = os.path.join(load_directory, f'user_attribute_dict_train.pkl')
    if not os.path.exists(load_path):
        print("User attribute dict not found in", load_directory)
        return None
    with open(load_path, 'rb') as f:
        user_attribute_dict = pickle.load(f)
    print("Loading user attribute dict from", load_directory)
    return user_attribute_dict

# self.data_name, self.mode,self.task, self.max_interactions, self.ask_num, self.alpha, self.change, self.neg_sample_num
def save_simulator_data(data, data_name, mode='train', task='EPM', turn=0, ask_num=2, alpha=0.5, change=0.1, neg_sample_num=50, load_epoch=0):
    save_directory = TMP_DIR[data_name] + f'/{task}_simulator_data/' 
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_file_name = f'simulator_data_{mode}_turn_{turn}_ask_{ask_num}_alpha_{alpha}_change_{change}_neg_{neg_sample_num}_epoch_{load_epoch}.pkl'
    save_path = os.path.join(save_directory, save_file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print("Saving {} data to {}".format(save_file_name, save_directory))

def load_simulator_data(data_name, mode='train', task='EPM', turn=0, ask_num=2, alpha=0.5, change=0.1, neg_sample_num=50, load_epoch=0):
    load_directory = TMP_DIR[data_name] + f'/{task}_simulator_data/'
    load_file_name = f'simulator_data_{mode}_turn_{turn}_ask_{ask_num}_alpha_{alpha}_change_{change}_neg_{neg_sample_num}_epoch_{load_epoch}.pkl'
    load_path = os.path.join(load_directory, load_file_name)
    if not os.path.exists(load_path):
        print("{} data not found in {}".format(load_file_name, load_directory))
        return None
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    print("Loading {} data from {}".format(load_file_name, load_directory))
    return data

def load_graph(data_name):
    save_directory = TMP_DIR[data_name]
    save_path = os.path.join(save_directory, 'retrieval_graph.pkl')
    with open(save_path, 'rb') as f:
        graph = dill.load(f)
    print("Loading retrieval graph from", save_directory)

    return graph

def load_embeddings(data_name):
    load_directory = TMP_DIR[data_name]
    entity_emebeds = np.load(os.path.join(load_directory, 'entity_embeds.npy'))
    
    relation_embeds = np.load(os.path.join(load_directory, 'relation_embeds.npy'))
    print(f"Loading entity embeddings from {load_directory}, shape: {entity_emebeds.shape}")
    print(f"Loading relation embeddings from {load_directory}, shape: {relation_embeds.shape}")
    return entity_emebeds, relation_embeds


def save_rl_agent(dataset, model, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-agent/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-agent/')
    torch.save(model, model_file)
    print('RL policy model saved at {}'.format(model_file))

def load_rl_agent(dataset, filename, epoch_user):
    model_file = TMP_DIR[dataset] + '/RL-agent/' + filename + '-epoch-{}.pkl'.format(epoch_user)
    model_dict = torch.load(model_file)
    print('RL policy model load at {}'.format(model_file))
    return model_dict

def save_turn_metric(dataset, filename, traing_step, testing_user, SR_turn_15):
    PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    with open(PATH, 'a') as f:
        f.write(f'===========Test Turn at Training Step:{traing_step}===============\n')
        f.write('Testing {} user tuples\n'.format(testing_user))
        for i in range(len(SR_turn_15)):
            f.write('Testing SR-turn@{}: {}\n'.format(i + 1, SR_turn_15[i]))
        f.write('================================\n')

def save_rl_metric(dataset, filename, epoch, metric_log, spend_time, mode='train'):
    PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user epochs\n'.format(epoch))
            f.write('training SR@5: {}\n'.format(metric_log[0]))
            f.write('training SR@10: {}\n'.format(metric_log[1]))
            f.write('training SR@15: {}\n'.format(metric_log[2]))
            f.write('training Avg@T: {}\n'.format(metric_log[3]))
            f.write('training hDCG: {}\n'.format(metric_log[4]))
            f.write('====Spending time: {}\n===='.format(spend_time))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))
    elif mode == 'test':
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR@5: {}\n'.format(metric_log[0]))
            f.write('Testing SR@10: {}\n'.format(metric_log[1]))
            f.write('Testing SR@15: {}\n'.format(metric_log[2]))
            f.write('Testing Avg@T: {}\n'.format(metric_log[3]))
            f.write('Testing hDCG: {}\n'.format(metric_log[4]))
            f.write('====Spending time: {}\n===='.format(spend_time))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))
def save_numpy_metric(dataset, filename, total_steps, formatted_metrics, final_formatted_SR_turn_15):
    result_dir = TMP_DIR[dataset] + '/RL-numpy-result/'
    os.makedirs(result_dir, exist_ok=True)
    
    np.save(f'{result_dir}metric-{filename}_step-{total_steps}', np.array(formatted_metrics))
    np.save(f'{result_dir}turn_metric-{filename}_step-{total_steps}', np.array(final_formatted_SR_turn_15))
    
    print(f'Successfully saved the evaluation result at step: {total_steps}!')


def save_rl_loss_log(dataset, filename, epoch, epoch_loss):
    PATH = TMP_DIR[dataset] + '/RL-log-merge/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/RL-log-merge/'):
        os.makedirs(TMP_DIR[dataset] + '/RL-log-merge/')
    with open(PATH, 'a') as f:
        f.write('===========Loss===============\n')
        f.write('Computing in {} epoch\n'.format(epoch))
        f.write('Actor loss : {}\n'.format(epoch_loss[0]))
        f.write('Critic loss : {}\n'.format(epoch_loss[1]))
        f.write('Entropy : {}\n'.format(epoch_loss[2]))


def write_args(data_name, args, file_dir, filename, args_name='Args', open_type='a'):
    if isinstance(args, dict):
        args_dict = args
        # open_type = 'a'
    else:
        args_dict = vars(args)
        # open_type = 'w'
    PATH = TMP_DIR[data_name] + f'/{file_dir}/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[data_name] + f'/{file_dir}/'):
        os.makedirs(TMP_DIR[data_name] + f'/{file_dir}/')
    with open(PATH, open_type) as f:
        pid = str(os.getpid())
        f.write(f'===== PID: {pid} ====\n')
        f.write(f'===== {args_name} Namespace ====\n')
        for k,v in args_dict.items():
            f.write(f'{k}={v}, ')
        f.write('\n')
        f.write('=============================\n')



def format_metrics(SR5, SR10, SR15, AvgT, Rank, total_reward, observe_num):
    metrics = [SR5, SR10, SR15, AvgT, Rank, total_reward]
    return [metric / observe_num for metric in metrics]

def format_logging(formatted_metrics, logging, total_steps, mode='train'):
    formatted_str = ', '.join(
        [f'SR{i * 5}:{metric}' for i, metric in enumerate(formatted_metrics[:3], 1)])
    # print(
    #     f'SR5:{formatted_metrics[0]}, SR10:{formatted_metrics[1]}, SR15:{formatted_metrics[2]}, AvgT:{formatted_metrics[3]}, Rank:{formatted_metrics[4]}, rewards:{formatted_metrics[5]} Observe_num:{args.observe_num}')
    logging.info(
        f'{mode} Step {total_steps}, {formatted_str}, AvgT:{formatted_metrics[3]}, Rank:{formatted_metrics[4]}, rewards:{formatted_metrics[5]}')
    


def format_writer(formatted_metrics, writer, total_steps, mode='train'):
    writer.add_scalar(f'Metric/{mode}-SR5',  formatted_metrics[0],  total_steps)
    writer.add_scalar(f'Metric/{mode}-SR10', formatted_metrics[1], total_steps)
    writer.add_scalar(f'Metric/{mode}-SR15', formatted_metrics[2], total_steps)
    writer.add_scalar(f'Metric/{mode}-AvgT', formatted_metrics[3], total_steps)
    writer.add_scalar(f'Metric/{mode}-Rank', formatted_metrics[4], total_steps)
    writer.add_scalar(f'Metric/{mode}-rewards', formatted_metrics[5], total_steps)


def dcg_at_k(clicked_indices, k):
    if not clicked_indices or k <= 0:
        return 0.0
    return sum([1.0 / np.log2(idx + 2) for idx in clicked_indices if idx < k])

def idcg_at_k(num_clicked_items, k):
    num_items = min(num_clicked_items, k)
    # return sum([1.0 / np.log2(np.arange(2, len(num_items) + 2))])
    return sum([1.0 / np.log2(i + 2) for i in range(num_items)])

def ndcg_at_k(clicked_indices, k):
    dcg = dcg_at_k(clicked_indices, k)
    idcg = idcg_at_k(len(clicked_indices), k)
    return dcg / idcg if idcg > 0 else 0




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")