

import os


LAST_FM_STAR = 'LAST_FM_STAR'
YELP_STAR = 'YELP_STAR'
BOOK = 'BOOK'

DATA_DIR = {
    LAST_FM_STAR: './data/lastfm_star',
    YELP_STAR: './data/yelp_star',
    BOOK: './data/book',
}
TMP_DIR = {
    LAST_FM_STAR: './tmp/lastfm_star',
    YELP_STAR: './tmp/yelp_star',
    BOOK: './tmp/book',
}



retrieval_graph_configs = {
        LAST_FM_STAR: {
            "entity_path": os.path.join(DATA_DIR[LAST_FM_STAR], "entities"),
            "node_datasets": {
                "items.csv": {"node_type": "item", "id_field": "map_id", "name_field": "name"},
                "users.csv": {"node_type": "user", "id_field": "map_id", "name_field": "name"},
                "attributes.csv": {"node_type": "attribute", "id_field": "map_id", "name_field": "name"},
            },
            "relation_path": os.path.join(DATA_DIR[LAST_FM_STAR], "relations"),
            "edge_datasets": {
                "user_item.csv": {"source_type": "user", "target_type": "item", "source_id_field": "user_id", "target_id_field": "item_id"},
                "item_attribute.csv": {"source_type": "item", "target_type": "attribute", "source_id_field": "item_id", "target_id_field": "attribute_id"},
                "user_user.csv": {"source_type": "user", "target_type": "user", "source_id_field": "user_id", "target_id_field": "user_id"},
                "user_attribute.csv": {"source_type": "user", "target_type": "attribute", "source_id_field": "user_id", "target_id_field": "attribute_id"},
                
            },
        },
        YELP_STAR: {
            "entity_path": os.path.join(DATA_DIR[YELP_STAR], "entities"),
            "node_datasets": {
                "items.csv": {"node_type": "item", "id_field": "map_id", "name_field": "name"},
                "users.csv": {"node_type": "user", "id_field": "map_id", "name_field": "name"},
                "attributes.csv": {"node_type": "attribute", "id_field": "map_id", "name_field": "name"},
            },
            "relation_path": os.path.join(DATA_DIR[YELP_STAR], "relations"),
            "edge_datasets": {
                "user_item.csv": {"source_type": "user", "target_type": "item", "source_id_field": "user_id", "target_id_field": "item_id"},
                "item_attribute.csv": {"source_type": "item", "target_type": "attribute", "source_id_field": "item_id", "target_id_field": "attribute_id"},
                "user_user.csv": {"source_type": "user", "target_type": "user", "source_id_field": "user_id", "target_id_field": "user_id"},
            },
        },
        BOOK: {
            "entity_path": os.path.join(DATA_DIR[BOOK], "entities"),
            "node_datasets": {
                "items.csv": {"node_type": "item", "id_field": "map_id", "name_field": "name"},
                "users.csv": {"node_type": "user", "id_field": "map_id", "name_field": "name"},
                "attributes.csv": {"node_type": "attribute", "id_field": "map_id", "name_field": "name"},
            },
            "relation_path": os.path.join(DATA_DIR[BOOK], "relations"),
            "edge_datasets": {
                "user_item.csv": {"source_type": "user", "target_type": "item", "source_id_field": "user_id", "target_id_field": "item_id"},
                "item_attribute.csv": {"source_type": "item", "target_type": "attribute", "source_id_field": "item_id", "target_id_field": "attribute_id"},
            },
        }
        
    }