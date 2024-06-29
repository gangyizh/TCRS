
from abc import ABC, abstractmethod
from typing import List, Any, Optional

import torch

class UserState(ABC):
    """
    User state class.
    Maintains and updates user state information, like preferences and interaction history.
    """
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize user state.
        """
        raise NotImplementedError

  

class UserBehaviorModel(ABC):
    """
    Abstract user behavior model class.
    Defines the interface for user reactions to recommended items and queried attributes.
    """

    @abstractmethod
    def decide_on_items(self, items: List[Any]) -> Any:
        """
        Decide response to a list of recommended items.
        :param items: List[Any] - List of recommended items.
        :return: Any - User response to recommended items.
        """
        raise NotImplementedError

    @abstractmethod
    def decide_on_attributes(self, attributes: List[str]) -> Any:
        """
        Decide response to a list of queried attributes.
        :param attributes: List[str] - List of queried attributes.
        :return: Any - User response to queried attributes.
        """
        raise NotImplementedError

    # @abstractmethod
    # def decide_on_language(self, language: str) -> Any:
    #     """
    #     Decide response to a language.
    #     :param language: str - Language.
    #     :return: Any - User response to language.
    #     """
    #     raise NotImplementedError




class PreferenceModel:
    """
    Preference Model class.
    Evaluates and scores user preferences.
    """
    @abstractmethod
    def estimate_preference(self) -> List:
        """
        Evaluate user preferences for items and attributes.
        :return: List - List of top K scored items & top K scored attributes.
        """
        # Logic to evaluate and score preferences
        raise NotImplementedError


class ScoringModel(ABC):
    """
    Abstract class for scoring items and attributes.
    """
    
    @abstractmethod
    def item_scores(self) -> torch.Tensor:
        """
        Computes the score for user-item pairs.

        """
        raise NotImplementedError

    @abstractmethod
    def attribute_scores(self) -> torch.Tensor:
        """
        Computes the score for user-attribute pairs.
        
        """
        raise NotImplementedError