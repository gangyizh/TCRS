# TCRS README
Paper: Reformulating Conversational Recommender Systems as Tri-Phase Offline Policy Learning


## Overview

TCRS (Tri-Phase Offline Policy Learning-based Conversational Recommendation System) introduces a novel architecture designed to reduce reliance on real-time user interactions and enhance the robustness and accuracy of CRS models by integrating model-based offline policy learning with a controllable user simulation.  Our open-source implementation aims to better understand and cater to individual user needs, leading to enhanced user experiences and increased adoption of CRS in real-world applications.


<!-- ## Prerequisites

Before you begin, ensure you have installed the following:

- Python >= 3.6
- Numpy >= 1.12
- PyTorch >= 1.0 -->


## Getting Started

To run the TCRS framework, follow these steps:

### 0. Build the Retrieval Graph:

First, you need to set up the graph retrieval structure to facilitate entity and relationship retrieval. Run the following command to execute `retrieval_graph.py`:

```bash
python retrieval_graph.py --data_name <dataset_name>
```

Replace `<dataset_name>` with the name of your dataset (e.g., `LAST_FM_STAR`, `YELP_STAR`, `BOOK`).


### 1. User Model Training

This initial phase involves training the Preference Estimation User Model (PEUM) that predicts user preferences using offline data. This model serves as the foundation for the subsequent policy learning stage. Execute the training with the following command:

```bash
python user_model_train.py --data_name <dataset_name> 
```

### 2. Reinforcement Learning Training & Evaluation
After training the user model, the next step is to use this model to train recommendation policies using the Proximal Policy Optimization (PPO) algorithm. This script includes both training and evaluation phases. The training phase optimizes the decision-making policy using the trained user model, while the evaluation phase assesses the policy's performance with a controllable user simulator.:

```bash
python RL_train.py --data_name <dataset_name> 
```


## Datasets

The TCRS framework supports multiple datasets, including `LAST_FM_STAR`, `YELP_STAR`, and `BOOK`. You can find the full version of these recommendation datasets via their respective sources.

## Code Structure

Here's a brief overview of the code structure:

```
TCRS/
│
├── data/                # Dataset directories
│   ├── book/
│   ├── lastfm/
│   └── yelp/
├── data_loader/         # Data loading modules
│   └── user_model_dataset.py
├── env/                 # Environment modules
│   ├── system_env.py     # Conversational Policy 
│   └── user_simulator.py # User Simulation
│
├── user_model/     # User modeling modules
│   └── preference_model.py  # Preference Estimation User Model (PEUM)
│   
│
├── rl_ppo/              # Reinforcement learning modules
│   ├── ppo_crs.py       # Actor-Critic PPO
│   ├── ppo_eval.py
│   └── replay_buffer.py
│
├── user_simulator/          # Basic User Simulator modules
│   ├── models.py
│   ├── user_behavior.py
│   └── user_state.py
│
├── config.py            # Configuration file
├── utils.py             # Utility functions
├── retrieval_graph.py   # Graph retrieval module
├── user_model_train.py  # PEUM training script
└── RL_train.py          # RL training script
```

<!-- ## Citation

If you find the TCRS framework useful in your research, please consider citing our paper:

```bibtex
@inproceedings{anonymous2024,
  title={Reformulating Conversational Recommender Systems as Tri-Phase Offline Policy Learning},
  author={Anonymous Author(s)},
  booktitle={},
  year={2024}
}
``` -->

<!-- ## License

The TCRS framework is released under the MIT License. See the LICENSE file for details. -->

<!-- ## Contact

For any questions or inquiries, please reach out to us at [xxx](mailto:xxx). We're here to help!

--- -->

Thank you for exploring the TCRS framework. Happy coding!