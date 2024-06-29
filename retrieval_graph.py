import argparse
import csv
import os
import networkx as nx
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
from collections import Counter, defaultdict

import utils
from config import *

class DataLoader(ABC):
    """Abstract class for loading different types of data files."""

    @abstractmethod
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        pass

class CSVDataLoader(DataLoader):
    """CSV file loader, implementing the DataLoader abstract class."""

    def __init__(self, delimiter: str = ','):
        """Initialize the CSVDataLoader with a specified delimiter.
        Args:
            delimiter (str): The delimiter to use for parsing CSV files. Defaults to ','.
        """
        self.delimiter = delimiter

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a CSV file.
        Args:
            file_path (str): The path to the CSV file.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the CSV rows.
        """
        data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter=self.delimiter)
                for row in reader:
                    data.append(row)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            raise FileNotFoundError
        except csv.Error as e:
            print(f"Error reading CSV file: {e}")
            raise csv.Error
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise Exception
        return data

class GraphBuilder:
    """Class for building and managing graphs."""

    def __init__(self):
        self.graph = nx.MultiGraph()  # Using MultiGraph
        self.node_lookup = {}  # For efficient node lookup
        self.node_name_lookup = {}  # Store node names
        self.node_counts = Counter()  # Track node counts
        self.edge_counts = Counter()  # Track edge counts
        self.node_ids_by_type = defaultdict(list)  # Track node map_ids by type
        self.connected_nodes_by_type = defaultdict(lambda: defaultdict(list))  # Track connected node map_ids by type



    def add_nodes(self, data: List[Dict[str, Any]], node_type: str, id_field: str='map_id', name_field: str = None) -> None:
        for item in data:
            node_id = f"{node_type}_{item[id_field]}"
            self.graph.add_node(node_id, type=node_type, **item)
            self.node_lookup[(node_type, item[id_field])] = node_id
            if name_field:
                self.node_name_lookup[node_id] = item[name_field]
            self.node_counts[node_type] += 1
            self.node_ids_by_type[node_type].append(int(item[id_field]))  # update node_ids_by_type

        

    def add_edges(self, data: List[Dict[str, Any]], source_type: str, target_type: str, source_id_field: str='source_id', target_id_field: str='target_id') -> None:
        for item in data:
            source_id = self.node_lookup.get((source_type, item[source_id_field]))
            target_id = self.node_lookup.get((target_type, item[target_id_field]))
            if source_id and target_id:
                self.graph.add_edge(source_id, target_id, **item)
                self.edge_counts[(source_type, target_type)] += 1
                # update connected_nodes_by_type
                self.connected_nodes_by_type[source_id][target_type].append(int(item[target_id_field]))
                self.connected_nodes_by_type[target_id][source_type].append(int(item[source_id_field]))
            
            else: # raise error if either source or target node is not found
                raise ValueError(f"Source or target node not found for edge: {item}")


    def load_nodes(self, loader: DataLoader, file_path: str, node_type: str, id_field: str, name_field: str = None) -> None:
        data = loader.load_data(file_path)
        self.add_nodes(data, node_type, id_field, name_field)
        print(f"Loaded {len(data)} nodes from {file_path}")
    

    def load_edges(self, loader: DataLoader, file_path: str, source_type: str, target_type: str, source_id_field: str, target_id_field: str) -> None:
        data = loader.load_data(file_path)
        self.add_edges(data, source_type, target_type, source_id_field, target_id_field)
        print(f"Loaded {len(data)} edges from {file_path}")


    def print_graph_info(self) -> None:
        print(f"Total number of nodes: {len(self.graph.nodes())}")
        print(f"Total number of edges: {len(self.graph.edges())}")

        # print(f"Total number of nodes: {self.graph.number_of_nodes()}")
        # print(f"Total number of edges: {self.graph.number_of_edges()}")

        for node_type, count in self.node_counts.items():
            print(f"Number of {node_type} nodes: {count}")
            print(f"Max-{node_type} id: {max(self.node_ids_by_type[node_type])}")
            
        for edge_type, count in self.edge_counts.items():
            print(f"Number of edges between {edge_type[0]} and {edge_type[1]}: {count}")

    def get_node(self, node_type_id_pair: Tuple[str, str]) -> Dict[str, Any]:
        node_id = self.node_lookup.get(node_type_id_pair)
        if node_id:
            return self.graph.nodes[node_id]
        else:
            return {}

    def get_node_name(self, node_type_id_pair: Tuple[str, str]) -> str:
        node_id = self.node_lookup.get(self._ensure_str_tuple(node_type_id_pair))
        return self.node_name_lookup.get(node_id, '')

    def get_node_count(self, node_type: str) -> int:
        return self.node_counts.get(node_type, 0)


    
    def get_edge(self, source_node_id_pair: Tuple[str, str], target_node_id_pair: Tuple[str, str]) -> Dict[str, Any]:
        source_node_id = self.node_lookup.get(self._ensure_str_tuple(source_node_id_pair)) 
        target_node_id = self.node_lookup.get(self._ensure_str_tuple(target_node_id_pair))
        return self.graph.get_edge_data(source_node_id, target_node_id, default={})
    
    def get_edge_count(self, source_type: str, target_type: str) -> int:
        return self.edge_counts.get((source_type, target_type), 0)

    def list_edges_for_node(self, node_type_id_pair: Tuple[str, str]):
        node_id = self.node_lookup.get(self._ensure_str_tuple(node_type_id_pair))
        if node_id is None:
            return []
        return list(self.graph.edges(node_id, data=True))
  

    def count_edges_for_node(self, node_type_id_pair: Tuple[str, str]):
        node_id = self.node_lookup.get(self._ensure_str_tuple(node_type_id_pair))
        if node_id is None:
            return 0
        return self.graph.degree(node_id)
    
    def get_ids_by_type(self, node_type: str) -> List[str]:
        return self.node_ids_by_type[node_type]

    def get_connected_ids_by_type(self, node_type_id_pair: Tuple[str, str], target_node_type: str) -> List[str]:
        node_id = self.node_lookup.get(self._ensure_str_tuple(node_type_id_pair))
        if node_id:
            return self.connected_nodes_by_type[node_id][target_node_type]
        else:
            return []

    def _ensure_str_tuple(self, pair: Tuple[Any, Any]) -> Tuple[str, str]:
        return tuple(str(x) for x in pair)

    
    
def main():

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=LAST_FM_STAR, choices=[LAST_FM_STAR, YELP_STAR, BOOK],
                        help='One of { LAST_FM_STAR, YELP_STAR, BOOK}.')
    args = parser.parse_args()
    data_name = args.data_name
    if data_name not in retrieval_graph_configs:
        print(f"Dataset {data_name} is not configured.")
        return

    config = retrieval_graph_configs[data_name]
    loader = CSVDataLoader(delimiter='\t')  # TODO delimiter
    builder = GraphBuilder()
    # Load and add nodes
    for entity_name, data_config in config["node_datasets"].items():
        entity_path = os.path.join(config["entity_path"], f"{entity_name}")
        builder.load_nodes(loader, entity_path, **data_config)
    # Load and add edges
    for relation_name, data_config in config["edge_datasets"].items():
        relation_path = os.path.join(config["relation_path"], f"{relation_name}")
        builder.load_edges(loader, relation_path, **data_config)
    utils.save_graph(builder, data_name)

if __name__ == "__main__":
    main()

