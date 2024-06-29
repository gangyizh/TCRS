
from itertools import chain, islice
import math
import numpy as np
import torch
from typing import Any, Optional, Tuple, List
from env.user_simulator import UserSimulator
from graph_init import GraphBuilder
from user_model.preference_model import EvolvingPreferenceModeling
from utils import cprint
class RecommendationEnv():
    """
    A reinforcement learning environment for conversational recommendation system.
    """
    def __init__(self, args, graph_builder: GraphBuilder, mode='train', ranking_model: EvolvingPreferenceModeling=None):
        """
        Initialize the environment.
        :param user_simulator: An instance of UserSimulator.
        :param state_model: An instance of ConversationState.
        :param max_turns: Maximum number of dialogue turns.


        """
        # == Parameters for conversation 
        self.mode = mode
        self.max_turns = args.max_turn

        # Init graph builder
        self.retrieval_graph = graph_builder
        
        # Init ranking model
        self.ranking_model = ranking_model

        # == Parameters for system interaction state 
        # Latent pool for user 
        self.ranked_latent_items = graph_builder.get_ids_by_type("item")  #TODO Ensure the ranking is maintained based on the preference model during the update stage
        self.ranked_latent_attributes = graph_builder.get_ids_by_type("attribute")  # all attributes
        # choice-based item pools : latent_items = click_intersection_item_pool | non-click union item pool | potiential item pool
        self.click_intersection_item_pool = set(self.ranked_latent_items)
        self.nonclick_union_item_pool = set()
        self.potiential_item_pool = set()
        # RL action pruning : Top-K item and attribute IDs  
        self.item_topk_ids = []
        self.attribute_topk_ids = []
       

        # == Parameters for system action prunning
        if self.mode == 'train':
            self.topk_item = args.topk_item
            self.topk_attribute = args.topk_attribute
        if self.mode == 'test':
            self.topk_item = args.test_topk_item
            self.topk_attribute = args.test_topk_attribute

        self.filter_strategy = args.filter_strategy

        self.reward_dict = {
            'ask_suc': 0.01, # Reward for successful user inquiry
            'ask_fail': -0.1, # Reward for failed user inquiry
            'rec_suc': 1, # Reward for successful recommendation
            'rec_fail': -0.1, # Reward for failed recommendation
            'max_turn': -0.3, # Penalty for reaching the maximum number of turns
            }

        

    
    def reset(self, user_simulator: UserSimulator) -> Tuple[Any, Any, Any, Any]:
        """
        Reset the environment to the initial state.
        :return: Current state, top-K item and attribute embeddings, and their IDs.
        """
        # Reset conversation state
        self.current_turn = 0
        
        # Get user' intent attribute and reset latent space
        self.user_simulator = user_simulator
        user_intent = self.user_simulator.user_state.get_intent_attribute() # list of int
        self._reset_latent_pool(user_intent) # reset latent item space
        self._reset_item_pool() # reset item pool :  self.click_intersection_item_pool, self.nonclick_union_item_pool, self.potiential_item_pool

        # == Score latent pool based on user's preference model & update topk item and attribute ids
        
        
        self._sorted_latent_item_and_attribute_pool()
        self.update_topk_item_ids(self.topk_item, mode=self.mode, initial_state=True)
        self.update_topk_attribute_ids(self.topk_attribute, mode=self.mode)
        # Get information for the state model  Re-sorted: self.item_topk_ids, self.attribute_topk_ids
        user_id, explicit_attributes_ids, item_topk_ids, attribute_topk_ids = self.get_state_info()

        # Init indictor
        self.indictor_dict = self.indictor_init()
        self.reward_turn_penalty = - 1
       

        
        return user_id, explicit_attributes_ids, item_topk_ids, attribute_topk_ids
    
    @property
    def action_space(self) -> Tuple[tuple, dict]:
        """
        Return the action space.
        :return: Tuple of action space.
        """
        item_space, attribute_space = len(self.ranked_latent_items), len(self.ranked_latent_attributes)
        self.action_space_dict = {"decision_space": 2, "item_space": item_space, "attribute_space": attribute_space}
        self.action_sapce = (item_space, attribute_space)
        return self.action_sapce, self.action_space_dict
    
    def indictor_init(self):
        
        indictor_dict = {
            "target_items": self.user_simulator.target_items,
            "target_item_attributes": self.user_simulator.target_item_attributes_set,
        }
        return indictor_dict

    def indictor_feedback(self, action_type: int, user_click_feedback: List[int], user_non_click_feedback: List[int]) -> float:
        """
        Get reward for an action.
        :param action_type: Type of action (0 for recommending items, 1 for asking attributes).
        :param selected_actions: List of selected actions.
        :return: Reward.
        """
        reward = 0
        done = False
        if action_type == 0:
            # Recommend items logic
            # Calculate reward
            if user_click_feedback:
                done = True
            reward = sum([self.item_reward_dict[item] for item in (user_click_feedback + user_non_click_feedback) if item != -1])

        elif action_type == 1:
            # Ask attributes logic
            # Calculate reward
            reward = sum([self.attribute_reward_dict[attribute] for attribute in (user_click_feedback + user_non_click_feedback) if attribute is not None])

        if self.current_turn >= self.max_turns:
            done = True

        # Reward for each turn
        reward += self.reward_turn_penalty

        return reward, done
    


    def world_model_step(self, action_type: int, selected_actions: List[int]) -> Tuple[Any, Any, Any, Any, float, bool, int]:
        """
        Perform an action and return the result.
        :param action_type: Type of action (0 for recommending items, 1 for asking attributes).
        :param selected_actions: List of selected actions.
        :return: Tuple of state, item and attribute embeddings, item and attribute IDs, reward, done, and rec_done index.
        """
        self.current_turn += 1
        self.action_type = action_type
        self.selected_actions = selected_actions

        rec_accept_inds = []

        if action_type == 0:
            # Recommend items logic
            
            rec_items, target_items = selected_actions, self.indictor_dict['target_items']
            if rec_items is None or not rec_items:
                # Handle empty or None recommended items
                raise ValueError("Recommended items cannot be empty or None.")
            accepted_items, rejected_items = list(set(rec_items) & set(target_items)), list(set(rec_items) - set(target_items))
            # Update latent item pool &  choice-based item pools
            self.update_latent_items(selected_actions)  # Remove recommended items from latent item pool
            self.update_item_pools_for_response(action_type, accepted_items, rejected_items) # update item pool
            
            if accepted_items: # Recommend successfully
                # Get the index of the first accepted item 
                rec_accept_inds = [selected_actions.index(item) for item in accepted_items]
            self.user_click_feedback, self.user_non_click_feedback = accepted_items, rejected_items 
            
        elif action_type == 1:
            # Ask attributes logic
            ask_attribuets, target_attributes = selected_actions, self.indictor_dict['target_item_attributes']

            click_attributes, non_click_attributes = list(set(ask_attribuets) & set(target_attributes)), list(set(ask_attribuets) - set(target_attributes))

            # Update latent attribute pool & choice-based item pools
            self.update_latent_attribute(selected_actions) # Remove queried attributes from latent attribute pool  
            self.update_item_pools_for_response(action_type, click_attributes, non_click_attributes) # update item pool   

            # Using or not using filter strategy
            if self.filter_strategy : #and self.mode == 'train' #  MCR setting:  SCPR, UNICORN, MCMIPL...
                self.filter_latent_space(click_attributes, non_click_attributes) # filter latent space
               
            self.user_click_feedback, self.user_non_click_feedback = click_attributes, non_click_attributes 
        

        # Get reward and done
        reward, done = self.indictor_feedback(action_type, self.user_click_feedback, self.user_non_click_feedback)

        # === Score latent pool & update topk item and attribute ids ===
        if action_type == 1:   # Ranking based on user's choice feedback
            self._sorted_latent_item_and_attribute_pool()
                
        self.update_topk_item_ids(self.topk_item, mode=self.mode)
        self.update_topk_attribute_ids(self.topk_attribute, mode=self.mode)
        # Get information for the state model
        user_id, explicit_attributes_ids, item_topk_ids, attribute_topk_ids = self.get_state_info()
        

        return user_id, explicit_attributes_ids, item_topk_ids, attribute_topk_ids, reward, done, rec_accept_inds


    def step(self, action_type: int, selected_actions: List[int]) -> Tuple[Any, Any, Any, Any, float, bool, int]:
        """
        Perform an action and return the result.
        :param action_type: Type of action (0 for recommending items, 1 for asking attributes).
        :param selected_actions: List of selected actions.
        :return: Tuple of state, item and attribute embeddings, item and attribute IDs, reward, done, and rec_done index.
        """
        self.current_turn += 1
        self.action_type = action_type
        self.selected_actions = selected_actions

        rec_accept_inds = []

        if action_type == 0:
            # Recommend items logic
            accepted_items, rejected_items = self.user_simulator.respond_to_recommendation(selected_actions)
            # Update latent item pool &  choice-based item pools
            self.update_latent_items(selected_actions)  # Remove recommended items from latent item pool
            self.update_item_pools_for_response(action_type, accepted_items, rejected_items) # update item pool
            
            if accepted_items: # Recommend successfully
                # Get the index of the first accepted item 
                rec_accept_inds = [selected_actions.index(item) for item in accepted_items]
            self.user_click_feedback, self.user_non_click_feedback = accepted_items, rejected_items 
            
        elif action_type == 1:
            # Ask attributes logic
            click_attributes, non_click_attributes = self.user_simulator.respond_to_query(selected_actions) # if selected_actions else ([], [-1])

            # Update latent attribute pool & choice-based item pools
            self.update_latent_attribute(selected_actions) # Remove queried attributes from latent attribute pool  
            self.update_item_pools_for_response(action_type, click_attributes, non_click_attributes) # update item pool   

            # Using or not using filter strategy
            if self.filter_strategy : #and self.mode == 'train' #  MCR setting:  SCPR, UNICORN, MCMIPL...
                self.filter_latent_space(click_attributes, non_click_attributes) # filter latent space
               
            self.user_click_feedback, self.user_non_click_feedback = click_attributes, non_click_attributes 
        

        # === Score latent pool & update topk item and attribute ids ===
        if action_type == 1:   # Ranking based on user's choice feedback
            self._sorted_latent_item_and_attribute_pool()
                
        self.update_topk_item_ids(self.topk_item, mode=self.mode)
        self.update_topk_attribute_ids(self.topk_attribute, mode=self.mode)
        # Get information for the state model
        user_id, explicit_attributes_ids, item_topk_ids, attribute_topk_ids = self.get_state_info()
        # Get reward and done


        reward, done = self.get_reward_and_done(action_type, self.user_click_feedback, self.user_non_click_feedback)

        return user_id, explicit_attributes_ids, item_topk_ids, attribute_topk_ids, reward, done, rec_accept_inds
    
    def get_state_embedding(self):
        """
        Get state embedding.
        :return: State embedding.
        """
        #user_ids, attribute_click_sequence, attribute_nonclick_sequence, cand_items, seq_click_lengths, seq_nonclick_lengths
        user_id, explicit_attributes_ids = self.user_simulator.get_user_and_explicit_attributes()
        record_non_choice_attributes = self.user_simulator.user_state.get_all_non_click_attribute()
        with torch.no_grad():
            state_embedding = self.ranking_model.conversation_state(torch.tensor([user_id], dtype=torch.long), torch.tensor([explicit_attributes_ids], dtype=torch.long), torch.tensor([record_non_choice_attributes], dtype=torch.long), torch.tensor([len(explicit_attributes_ids)], dtype=torch.long), torch.tensor([len(record_non_choice_attributes)], dtype=torch.long))  

        return state_embedding # Shape [1, state_embedding_size]
    
    def get_state_info(self):
        """
        Get state information for state model in RL.
        :return: Tuple of state, item and attribute embeddings, item and attribute IDs.
        """

        # Get information for the state model
        user_id, explicit_attributes_ids = self.user_simulator.get_user_and_explicit_attributes()
        # Get top-K item and attribute embeddings and their IDs

        return user_id, explicit_attributes_ids, self.item_topk_ids, self.attribute_topk_ids
    

    def get_reward_and_done(self, action_type: int, user_click_feedback: List[int], user_non_click_feedback: List[int]) -> float:
        """
        Get reward for an action.
        :param action_type: Type of action (0 for recommending items, 1 for asking attributes).
        :param selected_actions: List of selected actions.
        :return: Reward.
        """
        reward = 0
        done = False
        if action_type == 0:
            # Recommend items logic
            # Calculate reward
            if user_click_feedback:
                reward += self.reward_dict['rec_suc']
                done = True
            else:
                reward += self.reward_dict['rec_fail']

        elif action_type == 1:
            # Ask attributes logic
            # Calculate reward
            if user_click_feedback:
                # === Reward for successful user inquiry ===
                reward += self.reward_dict['ask_suc']
            if user_non_click_feedback:
                # === Reward for failed user inquiry ===
                reward += self.reward_dict['ask_fail']
        if self.current_turn >= self.max_turns:
            done = True
            reward = self.reward_dict['max_turn'] if self.current_turn >= self.max_turns else 0
        return reward, done

    
    def filter_latent_space(self, accept_attributes: List[int], reject_attributes: List[int]):
        """
        Filter latent space by accept and reject attributes.
        :param accept_attributes: List[int] - List of accepted attributes.
        :param reject_attributes: List[int] - List of rejected attributes.
        """
        if accept_attributes:   #Only treate accept attributes as filter
            accepted_items = set()
            for attribute in accept_attributes:
                accepted_items.update(self.retrieval_graph.get_connected_ids_by_type(("attribute", attribute), "item"))
            self.ranked_latent_items = list(set(self.ranked_latent_items).intersection(accepted_items))

            all_attributes = set()
            for item_id in self.ranked_latent_items:
                all_attributes.update(self.retrieval_graph.get_connected_ids_by_type(("item", item_id), "attribute"))
            self.ranked_latent_attributes = list(all_attributes - set(self.user_simulator.user_state.get_explicit_attributes()))

    def update_latent_items(self, recommended_items: List[int]):
        """
        Update latent item space.
        :param recommended_items: List[int] - List of recommended items.
        """
        recommended_items = set(recommended_items)
        self.ranked_latent_items = [item for item in self.ranked_latent_items if item not in recommended_items]

    def update_latent_attribute(self, query_attributes: List[int]):
        """
        Update latent attribute space.
        :param query_attributes: List[int] - List of queried attributes.
        """
        query_attributes = set(query_attributes)

        self.ranked_latent_attributes = [attribute for attribute in self.ranked_latent_attributes if attribute not in query_attributes]

    def update_item_pools_for_response(self, action_type: int , click_feedback: Optional[List[int]], non_click_feedback: Optional[List[int]]):
        """
        latent items = set(click intersection item pool) | set(non-click union item pool) | set(potiential item pool) 

        Update item pool for response.
        :param action_type: int - Type of action (0 for recommending items, 1 for asking attributes).
        :param click_attributes: Optional[List[str]] - List of click attributes.
        :param non_click_attributes: Optional[List[str]] - List of non-click attributes.
        """
        if  action_type == 0:
            self.click_intersection_item_pool = self.click_intersection_item_pool - set(click_feedback) - set(non_click_feedback)
            self.nonclick_union_item_pool = self.nonclick_union_item_pool - set(click_feedback) - set(non_click_feedback)
        if action_type == 1:

            if click_feedback:
                for attr in click_feedback:
                    self.click_intersection_item_pool = self.click_intersection_item_pool.intersection(set(self.retrieval_graph.get_connected_ids_by_type(("attribute", attr), "item")))
            if non_click_feedback:
                for attr in non_click_feedback:
                    self.nonclick_union_item_pool = self.nonclick_union_item_pool.union(set(self.retrieval_graph.get_connected_ids_by_type(("attribute", attr), "item")))
                self.nonclick_union_item_pool = self.nonclick_union_item_pool & set(self.ranked_latent_items)
        self.potiential_item_pool = set(self.ranked_latent_items) - self.nonclick_union_item_pool - self.click_intersection_item_pool


    def _get_latent_space(self):
        """
        Get latent space.
        :return: List[int] - Latent space.
        """
        return len(self.ranked_latent_items), len(self.ranked_latent_attributes)


    def render_init(self) -> None:
        cprint('=================Init Conversation=================')
        print(f'Init: User ID:{self.user_simulator.user} && Target Items:{self.user_simulator.target_items} && Intent:{self.user_simulator.user_state.get_intent_attribute()}')
        print(f'Init: Latent Items Space:{len(self.ranked_latent_items)} && Latent Attributes Space:{len(self.ranked_latent_attributes)}')

    
    def _sorted_latent_item_and_attribute_pool(self):
        latent_items, latent_attributes = self.get_latent_pool()
        with torch.no_grad():
            #user_ids, attribute_click_sequence, attribute_nonclick_sequence, cand_items, seq_click_lengths, seq_nonclick_lengths
            user_id, explicit_attributes_ids = self.user_simulator.get_user_and_explicit_attributes()
            record_non_choice_attributes = self.user_simulator.user_state.get_all_non_click_attribute()
            
            item_predictions, all_attribute_predictions = self.ranking_model(torch.tensor([user_id], dtype=torch.long), torch.tensor([explicit_attributes_ids], dtype=torch.long), torch.tensor([record_non_choice_attributes], dtype=torch.long), torch.tensor([latent_items], dtype=torch.long), torch.tensor([len(explicit_attributes_ids)], dtype=torch.long), torch.tensor([len(record_non_choice_attributes)], dtype=torch.long)) 
            

        

            latent_item_scores = item_predictions.squeeze(0).cpu() # Shape [latent_items]
            all_attribute_predictions = all_attribute_predictions.squeeze(0).cpu() # Shape [all_attributes]
            latent_attributes_scores = all_attribute_predictions[latent_attributes] # Shape [latent_attributes]

        
        latent_item_softmax_scores = torch.nn.functional.softmax(latent_item_scores, dim=0)
        latent_attributes_softmax_scores = torch.nn.functional.softmax(latent_attributes_scores, dim=0)

        # item_sorted_inds = torch.argsort(latent_item_softmax_scores, descending=True).tolist()
        # attribute_sorted_inds = torch.argsort(latent_attributes_softmax_scores, descending=True).tolist()
        
        sorted_item_score, sorted_item_inds = torch.sort(latent_item_softmax_scores, descending=True)
        sorted_attribute_score, sorted_attribute_inds = torch.sort(latent_attributes_softmax_scores, descending=True)
        
        item_sorted = [latent_items[ind] for ind in sorted_item_inds]
        attribute_sorted = [latent_attributes[ind] for ind in sorted_attribute_inds]

        self.item_reward_dict = dict(zip(item_sorted, sorted_item_score.tolist()))
        self.attribute_reward_dict = dict(zip(attribute_sorted, sorted_attribute_score.tolist()))
        self.ranked_latent_items = item_sorted
        self.ranked_latent_attributes = attribute_sorted
        
    
    def update_topk_item_ids(self, top_k_item: int = 10, mode='train',  initial_state=False):
        """
        Get top-K item IDs.
        :return: List[int] - Top-K item IDs.
        """
        # Latent item pool = click intersection item pool | non-click union item pool | potiential item pool
        if initial_state: # Get from sorted latent item pool  
            topk_item_pool = self.ranked_latent_items[:top_k_item]
        
       # get from all latent item pool
        topk_item_pool = self.ranked_latent_items[:top_k_item]
    
    
        self.item_topk_ids = topk_item_pool


    
    def update_topk_attribute_ids(self, top_k_attribute: int = 10, mode='train'):
        """
        Get top-K attribute IDs.
        """
        topk_attribute_pool = self.ranked_latent_attributes[:top_k_attribute]
        
        self.attribute_topk_ids = topk_attribute_pool


    def compute_current_precision(self):
        """
        Compute precision at each turn. (Precision = |A ∩ B| / |A|, where A is the target-item attributes and B is the user's adaptive preference.)
        :return: List[float] - Precision at each turn.
        """
        adaptive_preference = self.user_simulator.user_state.get_adaptive_preference()
        target_item_attributes = self.user_simulator.target_item_attributes_set
        pecision = len(set(adaptive_preference) & set(target_item_attributes)) / len(adaptive_preference)
        return pecision
    

    def size_current_item_pool(self):
        click_pool = set(self.user_simulator.target_items) & set(self.click_intersection_item_pool)
        nonclick_pool = set(self.user_simulator.target_items) & set(self.nonclick_union_item_pool)
        potiential_pool = set(self.user_simulator.target_items) & set(self.potiential_item_pool)

        return len(click_pool), len(nonclick_pool), len(potiential_pool)

    def get_latent_pool(self):
        """
        Get latent space.
        :return: List[int] - Latent space.
        """
        return self.ranked_latent_items, self.ranked_latent_attributes

    def _reset_latent_pool(self, intent: List):
        """
        Reset latent item space.
        :param intent: List[str] - List of intent attributes.
        """
        self.ranked_latent_items = [item for attribute in intent 
                                  for item in self.retrieval_graph.get_connected_ids_by_type(("attribute", attribute), "item")]
        self.ranked_latent_items = list(set(self.ranked_latent_items))

        self.ranked_latent_attributes =  list(set().union(*[self.retrieval_graph.get_connected_ids_by_type(("item", item_id), "attribute") for item_id in self.ranked_latent_items]))
        
        self.ranked_latent_attributes = list(set(self.ranked_latent_attributes) - set(intent))
        

    def _reset_item_pool(self):
        """
        Reset item pool.
        """
        self.click_intersection_item_pool = set(self.ranked_latent_items)
        self.nonclick_union_item_pool = set()
        self.potiential_item_pool = set()

    def render(self) -> None:
        """
        Render the environment to the screen.
        """
        
        cprint(f"======Current Turn: {self.current_turn}")
        if hasattr(self, 'action_type') and hasattr(self, 'selected_actions'):
            if self.action_type == 0:
                action_str = "Recommend Items" 
                
                print(f"System {action_str}: {self.selected_actions}")
                print(f"User Accept Items: {self.user_click_feedback}")
                print(f"User Reject Items: {self.user_non_click_feedback}")
                if self.user_click_feedback:
                    cprint(f"Recommend Successfully!")
            else:
                action_str =  "Ask Attributes"

                print(f"System {action_str}: {self.selected_actions}")
                print(f"User Accept Attributes: {self.user_click_feedback}")
                print(f"User Reject Attributes: {self.user_non_click_feedback}")
                if self.user_click_feedback:
                    cprint(f"Asking Successfully!")

        user_id, explicit_attributes_ids = self.user_simulator.get_user_and_explicit_attributes()
        intent_attributes = self.user_simulator.user_state.get_intent_attribute() 
        click_attributes, non_click_attributes = self.user_simulator.user_state.get_record_click_and_non_click_attributes()
        latent_item_space, latent_attribute_space = self._get_latent_space()


        def find_target_item_indices(pool, target_items, ranked_latent_items, self_find=False):
            """
            Find the indices of target items in a sorted pool.
            
            :param pool: A list of items representing the item pool.
            :param target_items: A list of target items to find indices for.
            :param ranked_latent_items: A list of items sorted based on certain criteria.
            :return: A list of indices representing the position of each target item in the sorted pool.
            """
            if self_find is False:
                # Filter the ranked_latent_items to include only those in the pool
                sorted_pool = [item for item in ranked_latent_items if item in pool]
            else:
                sorted_pool = ranked_latent_items

            # Create a dictionary mapping each item in the sorted pool to its index
            item_to_index = {item: idx for idx, item in enumerate(sorted_pool)}

            # Convert the sorted pool to a set for efficient lookup
            sorted_pool_set = set(sorted_pool)

            # Find the indices of target items in the sorted pool
            target_indices = [item_to_index[item] for item in target_items if item in sorted_pool_set]

            return target_indices
        
        print(f"===[Ground Truth in {self.mode}] mode] ")
        print(f"===Target Items {self.user_simulator.target_items} | Target_item Attributes: {self.user_simulator.target_item_attributes_set}===")
        print(f"===[User State]") 
        print(f"===Current State: User ID - {user_id}, Intent:{intent_attributes}, Explicit Attributes - {explicit_attributes_ids}===")
        print(f'===Current Evolving Preference: {self.user_simulator.user_state.get_adaptive_preference()}===')
        print(f'===Current Target Item Attributes Sample Rate : {self.user_simulator.a_target_sample_rate}, Personalized Sample Rate: {self.user_simulator.a_his_sample_rate}===')
        print(f"===Record Click Attributes: {click_attributes},  Non-Click Attributes: {non_click_attributes}===")
        print(f'===[System State]')
        print(f"===Current Latent Item Space: {latent_item_space},  Latent Attribute Space: {latent_attribute_space}===")
        print(f"===Current Click Item Pool Space: {len(self.click_intersection_item_pool)}, non-click Item Pool Space: {len(self.nonclick_union_item_pool)}, Potential Item Pool Space: {len(self.potiential_item_pool)}===")
        print(f"===Current Top-10 Items: {self.item_topk_ids[:10]},  Top-10 Attributes: {self.attribute_topk_ids[:10]}===")
        print(f'===Current Reward: {self.reward_dict}===')
        print(f'===[Model Prediction]')
        print(f'===Target items Ranking index in Latent Items: {find_target_item_indices(None, self.user_simulator.target_items, self.ranked_latent_items, self_find=True)}===')
        print(f'===Target Items Ranking index @ Click-Pool: {find_target_item_indices(self.click_intersection_item_pool, self.user_simulator.target_items, self.ranked_latent_items)} | @ Non-Click-Pool: {find_target_item_indices(self.nonclick_union_item_pool, self.user_simulator.target_items, self.ranked_latent_items)} | @Potiential-Pool: {find_target_item_indices(self.potiential_item_pool, self.user_simulator.target_items, self.ranked_latent_items)}===\n')

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass

