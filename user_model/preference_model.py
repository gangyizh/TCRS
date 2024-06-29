import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class EvolvingPreferenceModeling(nn.Module):
    def __init__(self, user_count, item_count, attribute_count, hidden_size, seq_model='gru', aggregation='attention', device='cuda'):
        super(EvolvingPreferenceModeling, self).__init__()

        self.device = device
        self.attribute_count = attribute_count
        # Initialize embeddings with padding idx
        self.user_embeds = nn.Embedding(user_count+1, hidden_size, padding_idx=user_count).to(device)
        self.item_embeds = nn.Embedding(item_count+1, hidden_size, padding_idx=item_count).to(device)
        self.attribute_embeds = nn.Embedding(attribute_count+1, hidden_size, padding_idx=attribute_count).to(device)
        

        self.hidden_size = hidden_size
        self.seq_model = seq_model
        self.aggregation = aggregation

        if seq_model == 'gru':
            self.seq_model_layer = nn.GRU(input_size=self.attribute_embeds.weight.data.shape[1],
                                          hidden_size=hidden_size,
                                          batch_first=True).to(device)
        elif seq_model == 'lstm':
            self.seq_model_layer = nn.LSTM(input_size=self.attribute_embeds.weight.data.shape[1] ,
                                          hidden_size=hidden_size,
                                          batch_first=True).to(device)
            


        elif seq_model == 'transformer':
            # self.pos_encoder = PositionalEncoding(user_embeds.size(1) + attribute_embeds.size(1))
            self.seq_model_layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.attribute_embeds.weight.data.shape[1], nhead=4, batch_first=True),
                num_layers=2).to(device)
    

        # Share MLP path

        self.system_ranking_path = nn.Sequential(
            nn.Linear(self.user_embeds.weight.data.shape[1] + hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ).to(device)

        # Attribute prediction
        self.attribute_prediction_layer = nn.Linear(hidden_size, attribute_count).to(device)
        # Item prediction
        # self.item_prediction_layer = nn.Linear(hidden_size, 1).to(device) 
    

    def forward(self, user_ids, attribute_click_sequence, attribute_nonclick_sequence, cand_items, seq_click_lengths, seq_nonclick_lengths):
        user_embedding = self.user_embeds(user_ids.to(self.device)) # Shape [bs, embedding_size]
        item_embeddings = self.item_embeds(cand_items.to(self.device)) # Shape [bs, max_cand_item_num, embedding_size] 
        click_attr_embeddings = self.attribute_embeds(attribute_click_sequence.to(self.device)) # Shape [bs, max_click_num, embedding_size]
        nonclick_attr_embeddings = self.attribute_embeds(attribute_nonclick_sequence.to(self.device)) # Shape [bs, max_nonclick_num, embedding_size]
    
        
        if self.seq_model in ['gru', 'lstm']:
            if click_attr_embeddings.sum() == 0 or attribute_click_sequence.size(1) == 0:
                click_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                packed_click_seq = torch.nn.utils.rnn.pack_padded_sequence(click_attr_embeddings, seq_click_lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, _ = self.seq_model_layer(packed_click_seq)  
                output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
                click_encoder_emb = output[:,-1,:] # Shape [bs, hidden_size]

            if nonclick_attr_embeddings.sum() == 0 or attribute_nonclick_sequence.size(1) == 0:
                nonclick_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                packed_nonclick_seq = torch.nn.utils.rnn.pack_padded_sequence(nonclick_attr_embeddings, seq_nonclick_lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, _ = self.seq_model_layer(packed_nonclick_seq)
                output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
                nonclick_encoder_emb = output[:,-1,:] # Shape [bs, hidden_size]you

        elif self.seq_model == 'transformer':
            if click_attr_embeddings.sum() == 0 or attribute_click_sequence.size(1) == 0:
                click_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                output = self.seq_model_layer(click_attr_embeddings) # Shape [bs, sequence_length, hidden_size]
                click_encoder_emb = torch.mean(output, dim=1) # Shape [bs, hidden_size]

           
            if nonclick_attr_embeddings.sum() == 0 or attribute_nonclick_sequence.size(1) == 0:
                nonclick_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                output = self.seq_model_layer(nonclick_attr_embeddings) # Shape [bs, sequence_length, hidden_size]
                nonclick_encoder_emb = torch.mean(output, dim=1) # Shape [bs, hidden_size]

        elif self.seq_model == 'mean':
            if click_attr_embeddings.sum() == 0 or attribute_click_sequence.size(1) == 0:
                click_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                click_encoder_emb = torch.mean(click_attr_embeddings, dim=1) # Shape [bs, hidden_size]
            if nonclick_attr_embeddings.sum() == 0 or attribute_nonclick_sequence.size(1) == 0:
                nonclick_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                nonclick_encoder_emb = torch.mean(nonclick_attr_embeddings, dim=1) # Shape [bs, hidden_size]

        
        # Combine user embeddings click sequence embeddings amd nonclick sequence embeddings
        combined = torch.cat((user_embedding, click_encoder_emb, nonclick_encoder_emb), dim=1) # Shape [bs, user_emb_size+hidden_size+hidden_size]

        # Through the shared MLP path
        shared_output = self.system_ranking_path(combined) # Shape [bs, hidden_size]

        # Attribute prediction
       
        attribute_predictions = self.attribute_prediction_layer(shared_output) # Shape [bs, num_attributes]


        # Item prediction
        
        item_predictions = torch.bmm(shared_output.unsqueeze(1), item_embeddings.transpose(1, 2)).squeeze(1) # Shape [bs, max_cand_item_num]
       
        return item_predictions, attribute_predictions
    

    def conversation_state(self, user_ids, attribute_click_sequence, attribute_nonclick_sequence, seq_click_lengths, seq_nonclick_lengths):

        user_embedding = self.user_embeds(user_ids.to(self.device)) # Shape [bs, embedding_size]
        click_attr_embeddings = self.attribute_embeds(attribute_click_sequence.to(self.device)) # Shape [bs, max_click_num, embedding_size]
        nonclick_attr_embeddings = self.attribute_embeds(attribute_nonclick_sequence.to(self.device)) # Shape [bs, max_nonclick_num, embedding_size]
        
        if self.seq_model in ['gru', 'lstm']:
            if click_attr_embeddings.sum() == 0 or attribute_click_sequence.size(1) == 0:
                click_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                packed_click_seq = torch.nn.utils.rnn.pack_padded_sequence(click_attr_embeddings, seq_click_lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, _ = self.seq_model_layer(packed_click_seq)  
                output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
                click_encoder_emb = output[:,-1,:] # Shape [bs, hidden_size]

            if nonclick_attr_embeddings.sum() == 0 or attribute_nonclick_sequence.size(1) == 0:
                nonclick_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                packed_nonclick_seq = torch.nn.utils.rnn.pack_padded_sequence(nonclick_attr_embeddings, seq_nonclick_lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_output, _ = self.seq_model_layer(packed_nonclick_seq)
                output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
                nonclick_encoder_emb = output[:,-1,:] # Shape [bs, hidden_size]you

        elif self.seq_model == 'transformer':
            if click_attr_embeddings.sum() == 0 or attribute_click_sequence.size(1) == 0:
                click_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                output = self.seq_model_layer(click_attr_embeddings) # Shape [bs, sequence_length, hidden_size]
                click_encoder_emb = torch.mean(output, dim=1) # Shape [bs, hidden_size]

            if nonclick_attr_embeddings.sum() == 0 or attribute_nonclick_sequence.size(1) == 0:
                nonclick_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                output = self.seq_model_layer(nonclick_attr_embeddings) # Shape [bs, sequence_length, hidden_size]
                nonclick_encoder_emb = torch.mean(output, dim=1) # Shape [bs, hidden_size]

        elif self.seq_model == 'mean':
            if click_attr_embeddings.sum() == 0 or attribute_click_sequence.size(1) == 0:
                click_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                click_encoder_emb = torch.mean(click_attr_embeddings, dim=1) # Shape [bs, hidden_size]
            if nonclick_attr_embeddings.sum() == 0 or attribute_nonclick_sequence.size(1) == 0:
                nonclick_encoder_emb = torch.zeros(user_embedding.size()).to(self.device)
            else:
                nonclick_encoder_emb = torch.mean(nonclick_attr_embeddings, dim=1) # Shape [bs, hidden_size]

        
        # Combine user embeddings click sequence embeddings amd nonclick sequence embeddings
        combined = torch.cat((user_embedding, click_encoder_emb, nonclick_encoder_emb), dim=1) # Shape [bs, user_emb_size+hidden_size+hidden_size]

        # Through the shared MLP path
        shared_output = self.system_ranking_path(combined) # Shape [bs, hidden_size]
        
        return shared_output
    