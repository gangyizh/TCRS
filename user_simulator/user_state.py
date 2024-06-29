
from typing import Any, List, Optional, Tuple
from graph_init import GraphBuilder
from user_model.models import UserState

class ChoiceUserState(UserState):
    """
    User state class.
    Maintains and updates user state information, like preferences and interaction history.
    """
    def __init__(self):  
        
        
        self.initial_intent_attribute: List[int] = []

        self.adaptive_preference: List[int] = []
        self.record_preference: List[List[int]] = []

        self.current_click_attribute: List[int] = []
        self.current_non_click_attribute: List[int] = []
        
        self.record_click_attribute: List[List[int]] = []
        self.record_non_click_attribute: List[List[int]] = []

    def init_intent_state(self, intent: List):
        """
        User usually have a clear initial intent preferences when initiating a conversation.
        Initialize intent.
        :param intent: List[str] - List of intent attributes.
        """
        self.initial_intent_attribute = intent


    def update_choice_state(self, click_attribute: List[int], non_click_attribute: List[int]):
        """
        Update state based on user feedback.
        :param feedback: Any - User feedback information.
        :param click: bool - Whether the user clicked on the attributes.
        """
        if click_attribute:
            self.current_click_attribute = click_attribute
            self.record_click_attribute.append(click_attribute)
        if non_click_attribute:
            self.current_non_click_attribute = non_click_attribute
            self.record_non_click_attribute.append(non_click_attribute)

    

    def update_preference(self, preference: List[int]):
        """
        Update preference.
        :param preference: List[int] - User preference information.
        """
        self.adaptive_preference = preference
        self.record_preference.append(preference)
    

    def get_adaptive_preference(self):
        """
        Get current preference.
        :return: List[int] - Current preference.
        """
        return self.adaptive_preference
    
    def get_intent_attribute(self):
        """
        Get intent attribute.
        :return: List[int] - Intent attribute.
        """
        return self.initial_intent_attribute
    
    def get_all_non_click_attribute(self):
        """
        Get all non click attribute.
        :return: List[int] - All non click attribute.
        """
        return [attr for sublist in self.record_non_click_attribute for attr in sublist]
    
    def get_explicit_attributes(self):
        """
        Get explicit preference: initial intent + record click attribute.
        :return: List[int] - Explicit preference.
        """
        intent_attributes = self.initial_intent_attribute # list
        click_attributes = [item for sublist in self.record_click_attribute for item in sublist] # list 

        return intent_attributes + click_attributes # list

    def get_record_click_and_non_click_attributes(self):
        """
        Get record click attribute and record non click attribute.
        :return: List[int] - Record click attribute and record non click attribute.
        """
        return self.record_click_attribute, self.record_non_click_attribute
    def get_current_click_and_non_click_attributes(self):
        """
        Get current click attribute and current non click attribute.
        :return: List[int] - Current click attribute and current non click attribute.
        """
        return self.current_click_attribute, self.current_non_click_attribute
    
