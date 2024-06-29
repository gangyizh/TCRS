from itertools import chain
import math
from typing import Any, List, Optional, Tuple, Union
import numpy as np

import torch
from graph_init import GraphBuilder

from user_simulator.user_state import ChoiceUserState
from user_simulator.user_behavior import ChoiceBehavior
from utils import load_simulator_user_attribute_dict



class UserSimulator:
    """
    User simulator class.
    Simulates user behavior and responses.
    """

    def __init__(self, ui_pair: Tuple[int, Optional[Union[int, list]]], args, graph_builder: GraphBuilder,  mode='train'):
        assert mode in ['train', 'test'] #, "mode must be 'train' or 'test'"
        # Init graph builder
        self.retrieval_graph = graph_builder

        # Single item or Multiple items based on the training dataset
        if isinstance(ui_pair[1], int):
            self.user, self.target_items = ui_pair[0], [ui_pair[1]]
        else:
            self.user, self.target_items = ui_pair

        
        # Init user state, behavior model and preference model
        user_state = ChoiceUserState()
        behavior_model = ChoiceBehavior()


        self.user_state = user_state
        self.behavior_model = behavior_model
        self.mode = mode


        
        # Parameters for Init intent to start conversation
        self.intent_num = args.intent_num

        # Sample-based simulation
        
        # Based on the training dataset, But in the test dataset, we need to load the user attribute counters from the training dataset .
        user_attribute_counters = load_simulator_user_attribute_dict(args.data_name, mode='train')
        
        self.target_item_attributes_set = set().union(*[self.retrieval_graph.get_connected_ids_by_type(("item", item_id), "attribute") for item_id in self.target_items])
        self.his_attributs_set = set(user_attribute_counters[str(self.user)])
        self.attribute_counter = user_attribute_counters[str(self.user)]
        
        # Parameters for rule-based simulation
        self.target_alpha = 1-args.personalization_alpha
        self.adaptive_change = args.adaptive_change
        self.target_sample_strategy = args.target_sample_strategy
        self.his_sample_strategy = args.his_sample_strategy
        self.user_prefer_num = len(self.target_item_attributes_set)
        self.a_target_sample_rate, self.a_his_sample_rate = self.target_alpha, 1-self.target_alpha


 
        # == Init intent to start conversation ==
        self.init_intent_to_start_conversation()
        

    def init_intent_to_start_conversation(self):
        """
        Reset user intent to start a conversation.
        """
        
        #===
        intent = list(np.random.choice(list(self.target_item_attributes_set), size=self.intent_num, replace=False))
        #===
        # update user state
        self.user_state.init_intent_state(intent)  #  initialize intent state 
        adaptive_preference = self.update_user_adaptive_preference(initial=True) # update adaptive preference for items
        self.user_state.update_preference(adaptive_preference) # update user preferemce state

    def respond_to_query(self, query_attributes: Optional[List[int]]) -> (List[int], List[int]):
        """
        Simulate user response to queried attributes.
        :param query_attributes: Optional[List[str]] - List of queried attributes.
        :return: (List[int], List[int]) - choice_attributes, non_choice_attributes.
        """
        if query_attributes is None or not query_attributes:
            # Handle empty or None query attributes
            # raise ValueError("Query attributes cannot be empty or None.")
            print("Query attributes be empty or None.")


        # current_preference = list(self.target_item_attributes_set)  #TODO clear preference in some item—centric CRS setting：  SCPR, UNICORN, MCMIPL ...
        current_preference = self.user_state.get_adaptive_preference()

        choice_attributes, non_choice_attributes = self.behavior_model.decide_on_attributes(query_attributes, current_preference)

        # update user state
        self.user_state.update_choice_state(choice_attributes, non_choice_attributes)  # update user choice state 
        adaptive_preference = self.update_user_adaptive_preference() # update adaptive preference 
        self.user_state.update_preference(adaptive_preference) # update user preferemce state
        return choice_attributes, non_choice_attributes
    
    
        
    
    def respond_to_recommendation(self, recommended_items: Optional[List[Any]]) -> List[int]:
        """
        Simulate user response to recommended items.
        :param recommended_items: Optional[List[Any]] - List of recommended items.
        :return: List[int] - accept_items .
        """
        if recommended_items is None or not recommended_items:
            # Handle empty or None recommended items
            raise ValueError("Recommended items cannot be empty or None.")
        accept_items, reject_items = self.behavior_model.decide_on_items(recommended_items, self._get_target_items()) # list
        
        return accept_items, reject_items
    
    
    


    def update_user_adaptive_preference(self, initial=False):
        """
        Update user adaptive preference.
        """
        # == Initial User Preference ==
        user_confirm_attributes = self.user_state.get_explicit_attributes() # list
        prefer_sample_num = max((self.user_prefer_num - len(user_confirm_attributes)), 0)
        if initial:
            a_target_sample_num, a_his_sample_num = math.ceil(prefer_sample_num * self.a_target_sample_rate), math.ceil(prefer_sample_num * self.a_his_sample_rate)
        else: # Feeback from user to update sample rate
            # Update sample rate for target and historical attributes
            turn_click_attrs, turn_nonclick_attrs = self.user_state.get_current_click_and_non_click_attributes()
            a_target_click = set(turn_click_attrs) & self.target_item_attributes_set
            a_target_nonclick = set(turn_nonclick_attrs) & self.target_item_attributes_set
            turn_change_rate = 0 if len(a_target_click) == len(a_target_nonclick) else (self.adaptive_change if len(a_target_click) > len(a_target_nonclick) else -self.adaptive_change)
            self.a_target_sample_rate = max(0, min(1, self.a_target_sample_rate + turn_change_rate))
            self.a_his_sample_rate = 1 - self.a_target_sample_rate
            a_target_sample_num, a_his_sample_num = math.ceil(prefer_sample_num * self.a_target_sample_rate), math.ceil(prefer_sample_num * self.a_his_sample_rate)
        
        # Target Preference:  inverse weighted sample strategy 
        if self.target_sample_strategy == 'inverse_weighted':
            valid_a_target_prefer = list(self.target_item_attributes_set - set(user_confirm_attributes))
            valid_a_target_prefer_weight = np.array([1/self.attribute_counter.get(a, 1) for a in valid_a_target_prefer])
            valid_a_target_prefer_weight = valid_a_target_prefer_weight/valid_a_target_prefer_weight.sum()
            turn_target_prefer = np.random.choice(list(valid_a_target_prefer), size=a_target_sample_num, replace=False, p=valid_a_target_prefer_weight) if len(valid_a_target_prefer) else []
        elif self.target_sample_strategy == 'weighted':
            valid_a_target_prefer = list(self.target_item_attributes_set - set(user_confirm_attributes))
            valid_a_target_prefer_weight = np.array([self.attribute_counter.get(a, 1) for a in valid_a_target_prefer])
            valid_a_target_prefer_weight = valid_a_target_prefer_weight/valid_a_target_prefer_weight.sum()
            turn_target_prefer = np.random.choice(list(valid_a_target_prefer), size=a_target_sample_num, replace=False, p=valid_a_target_prefer_weight) if len(valid_a_target_prefer) else []
        elif self.target_sample_strategy == 'uniform':
            turn_target_prefer = np.random.choice(list(self.target_item_attributes_set - set(user_confirm_attributes)), size=a_target_sample_num, replace=False) if len(self.target_item_attributes_set - set(user_confirm_attributes)) else []
        else:
            raise ValueError("target_sample_strategy must be 'inverse_weighted' or 'weighted' or 'uniform'.")
        
        
        # Historical Preference: weighted sample strategy
        if self.his_sample_strategy == 'weighted':
            valid_a_his_prefer = list(self.his_attributs_set - set(user_confirm_attributes))
            valid_a_his_prefer_weight = np.array([self.attribute_counter.get(a, 1) for a in valid_a_his_prefer])
            valid_a_his_prefer_weight = valid_a_his_prefer_weight/valid_a_his_prefer_weight.sum()
            turn_his_prefer = np.random.choice(list(valid_a_his_prefer), size=min(a_his_sample_num, len(valid_a_his_prefer)), replace=False, p=valid_a_his_prefer_weight) if len(valid_a_his_prefer) else []
        elif self.his_sample_strategy == 'uniform':
            valid_a_his_prefer = list(self.his_attributs_set - set(user_confirm_attributes))
            turn_his_prefer = np.random.choice(list(valid_a_his_prefer), size=min(a_his_sample_num, len(valid_a_his_prefer)), replace=False) if len(valid_a_his_prefer) else []
        else:
            raise ValueError("his_sample_strategy must be 'weighted' or 'uniform'.")
        
        # Update user's preference label
        adaptive_preference = list(turn_target_prefer) + list(turn_his_prefer) + user_confirm_attributes

        return adaptive_preference # list

        
    
    
    
    def _get_target_items(self):
        """
        Get target items.
        :return: List[int] - Target items.
        """
        return self.target_items
    
    # user_embedding, attribute_embeddings = self.user_simulator.get_user_and_explicit_attributes_embedding()
    def get_user_and_explicit_attributes(self):
        """
        Get user and explicit attributes embeddings.
        :return: Tuple[torch.Tensor, torch.Tensor] - User and explicit attributes embeddings.
        """

        explicit_attributes = self.user_state.get_explicit_attributes() 

        return self.user, explicit_attributes
