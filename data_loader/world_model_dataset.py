import copy
import math
import random
from threading import Timer
import time
import numpy as np
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  
from multiprocessing import Pool
from utils import load_simulator_data, save_simulator_data, save_simulator_user_attribute_dict,load_simulator_user_attribute_dict

class UserSimulatorDataset(Dataset):
    def __init__(self, data_name, user_item_dict, item_attribute_dict, neg_sample_num=50, mode='train', num_workers=32):
        self.data_name = data_name
        self.mode = mode
        self.task = "EPM"
        self.num_workers = num_workers
        self.neg_sample_num = neg_sample_num
        print("Mode:{} Dataloader task: system_ranking".format(mode))
        print("Neg sample num: {}".format(neg_sample_num))
    
        self.user_item_pairs = [(str(u), str(i)) for u, items in user_item_dict.items() for i in items]

        if self.mode == 'test': # For test, we only use 5000 user-item pairs
            self.user_item_pairs = random.sample(self.user_item_pairs, 2500)

        self.user_item_dict = user_item_dict
        self.item_attribute_dict = item_attribute_dict

        # check empty items attributes
        empty_items = [item for item, attrs in item_attribute_dict.items() if len(attrs) == 0]
        print("Empty attributes items: ",  empty_items)
        
        

        if self.mode == 'train':
            self.user_attribute_counters = {user: Counter([attr for item in items for attr in item_attribute_dict.get(str(item), [])]) for user, items in user_item_dict.items()}
            save_simulator_user_attribute_dict(data_name, self.user_attribute_counters, self.mode)
        elif self.mode == 'test':
            # Based on the training dataset, But in the test dataset, we need to load the user attribute counters from the training dataset . We need to ensure historical attributes are equal or larger than the target item attributes.
            self.user_attribute_counters = load_simulator_user_attribute_dict(data_name, self.mode)
            
        self.all_attributes = set([attr for attrs in item_attribute_dict.values() for attr in attrs])
        print("UserSimulatorDataset: All attributes: {}".format(len(self.all_attributes)))
        self.all_items = set(map(int, item_attribute_dict.keys())) # 
        self.data = []
        self.attribute_padding = max(self.all_attributes) + 1


    def generate_interaction_single(self, user_item_pair):
        """
        user: str
        target_item: str
        """
        user, target_item = user_item_pair
        # a_target = set(self.item_attribute_dict.get(target_item, [])) # Target item attributes
        # a_his = set(self.user_attribute_counters[user]) # Historical attributes
        # attribute_counter = self.user_attribute_counters[user]
        def sample_turn_geometric(p, max_turns):
            # sample the number of interactions for the user
            turn = np.random.geometric(p)
            return min(turn, max_turns)  
        max_turns = 15
        turn_prob = 0.35
        ask_num = 2
        

        target_item_attributes = set(self.item_attribute_dict.get(target_item, [])) # Target item attributes
        asked_attirubtes_num = sample_turn_geometric(turn_prob, max_turns) * ask_num
        sample_attributes_num = min(asked_attirubtes_num, len(target_item_attributes))   

        if len(target_item_attributes) <= 1:
            prefer_attributes, unprefer_attributes, predict_attributes = [], [], list(target_item_attributes)
        else:
            # sample_attributes_num = random.randint(1, len(target_item_attributes)-1)
            prefer_sample_num = random.randint(1, sample_attributes_num)
            unprefer_sample_num = sample_attributes_num - prefer_sample_num
            prefer_attributes = random.sample(list(target_item_attributes), prefer_sample_num)
            unprefer_attributes = random.sample(list(target_item_attributes - set(prefer_attributes)), unprefer_sample_num)

            predict_attributes = list(target_item_attributes - set(prefer_attributes) - set(unprefer_attributes))
        
        
        if  self.mode == 'train':
            # Negative sampling for system ranking
            interacted_items = set(self.user_item_dict[user])
            # pos_interacted_items = {item for item in interacted_items if set(click_attributes).issubset(set(self.item_attribute_dict[str(item)]))}
            neg_items_all = list(self.all_items - {int(target_item)} - interacted_items)
            neg_items_sample = random.sample(neg_items_all, self.neg_sample_num)
            cand_items = [int(target_item)] + neg_items_sample
            item_labels = [1] + [0] * self.neg_sample_num
        elif  self.mode == 'test':
            neg_items_all = list(self.all_items - {int(target_item)})
            neg_items_sample = random.sample(neg_items_all, self.neg_sample_num)
            cand_items = [int(target_item)] + neg_items_sample
            item_labels = [1] + [0] * self.neg_sample_num

        
        return (np.int32(user),
            np.int32(target_item),
            np.array(prefer_attributes, dtype=np.int32), 
            np.array(unprefer_attributes, dtype=np.int32), 
            np.array(predict_attributes, dtype=np.int32), 
            np.array(cand_items, dtype=np.int32), 
            np.array(item_labels, dtype=np.int32),
            )
        
    def generate_interaction(self, load_from_file=False, load_epoch=0):

        if load_from_file:
            load_data = load_simulator_data(self.data_name, self.mode, self.task, 0,0,0,0, self.neg_sample_num, load_epoch=load_epoch)
            if load_data is not None:
                self.data = load_data
                return
            else:
                print("No data found in, generating new data...")

        # Start to generate data
        self.data = []
        # with Timer(print_tmpl='Pool() takes {:.1f} seconds'):
        print("Dataset length: {}".format(len(self.user_item_pairs)))
        start_time = time.time()
        with Pool(self.num_workers) as p:
            self.data = p.map(self.generate_interaction_single, self.user_item_pairs)
        print("Pool{} takes {:.1f} seconds to generate data".format(self.num_workers, time.time() - start_time))

        # Save simulator data
        if load_from_file:
            save_simulator_data(self.data, self.data_name, self.mode,self.task, 0,0,0,0, self.neg_sample_num, load_epoch=load_epoch)

   
            
    def __len__(self):
        return len(self.user_item_pairs)
    
    def __getitem__(self, idx):
        """
        user: int
        target_item: int
        click_attributes: list
        nonclick_attributes: list
        a_target_list: list
        cand_items: list
        item_labels: list
        """
        
        # assert len(self.data) > 0, "Data is empty. Please call __generate_interaction() first."
      

        user, target_item, click_attributes, nonclick_attributes, user_prefer_label, cand_items, item_labels = self.data[idx]
        # Convert data to PyTorch tensors or appropriate format here
        
        return user, target_item, click_attributes, nonclick_attributes, user_prefer_label, cand_items, item_labels

