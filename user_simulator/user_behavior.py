
from typing import Any, List, Optional, Tuple
from user_model.models import UserBehaviorModel

class ChoiceBehavior(UserBehaviorModel):
    """
    Choice behavior model class.
    Simulates user behavior in choice-based interactions.
    """
    
    def decide_on_items(self, items: List[Any], target_items: List[any]) -> Any:
        #  choice items based on target_items
        return list(set(items) & set(target_items)), list(set(items) - set(target_items))
    
    def decide_on_attributes(self, attributes: List[str], current_preference: List[any]) -> Any:
        #  choice attributes based on current preference
        choice_attributes = list(set(attributes) & set(current_preference))
        non_choice_attributes = list(set(attributes) - set(choice_attributes))
        return choice_attributes, non_choice_attributes