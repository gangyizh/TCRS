
import logging
import time

from tqdm import tqdm
from data_loader.user_model_dataset import UserSimulatorDataset
from torch.utils.data import DataLoader
from user_model.preference_model import EvolvingPreferenceModeling
import argparse
from utils import *
from torch.nn.utils.rnn import pad_sequence

def calculate_hit_at_k(predictions, true_labels, K):
    """
    Calculate the value of Hit@K.

    Args:
    predictions (torch.Tensor): A tensor of shape (batch_size, item_num), where each row is a logit vector.
    true_labels (torch.Tensor): A tensor of shape (batch_size,) representing the true labels for each sample.
    K (int): The number of predictions to consider.
    """
    assert predictions.size(0) == len(true_labels)
    
    top_k_preds = predictions.topk(K, dim=1).indices
    match = top_k_preds == true_labels.view(-1, 1)
    hits = match.sum().item()
    return hits


def calculate_ndcg_at_k(predictions, true_labels, K):
    """
    Calculate the value of NDCG@K.

    Args:
    predictions (torch.Tensor): A tensor of shape (batch_size, item_num), where each row is a logit vector.
    true_labels (torch.Tensor): A tensor of shape (batch_size,) representing the true labels for each sample.
    K (int): The number of predictions to consider.

    Returns:
    float: The value of NDCG@K.
    """
    # Get the top K predictions for each row (sample)
    top_k_preds = predictions.topk(K, dim=1).indices
    
    # Calculate DCG
    discounts = torch.log2(torch.arange(2.0, K + 2.0)).to(predictions.device)
    dcg = ((top_k_preds == true_labels.view(-1, 1)).float() / discounts).sum(dim=1)

    # Calculate IDCG
    idcg = 1.0  # Only one positive label per sample
    ndcg = dcg.sum().item()

    return ndcg



def calculate_precision_recall_f1(predictions, true_label_inds):
    """
    Calculate the values of Precision, Recall, and F1.

    Args:
    predictions (torch.Tensor): A tensor of shape (batch_size, item_num), where each row is a logit vector.
    true_label_inds (torch.Tensor): A tensor of shape (batch_size, label_inds_max_length) representing the indices of the true labels for each sample.

    Returns:
    tuple: A tuple of floats (precision, recall, f1).
    """
    # 
    pred_top_inds = torch.topk(predictions, true_label_inds.size(1)).indices
    precision_sum, recall_sum, f1_sum = 0.0, 0.0, 0.0
    for i in range(predictions.size(0)):
        valid_mask = true_label_inds[i,: ] != predictions.size(1) # Padding value is attribute_count
        pred_valid_inds = pred_top_inds[i,: ][valid_mask].tolist()
        true_label_valid_inds = true_label_inds[i,: ][valid_mask].tolist()
        TP = len(set(pred_valid_inds) & set(true_label_valid_inds))
        FP = len(set(pred_valid_inds) - set(true_label_valid_inds))
        FN = len(set(true_label_valid_inds) - set(pred_valid_inds))

        # Compute: Precision, Recall
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        # Compute: F1
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    return  precision_sum, recall_sum, f1_sum


def custom_collate_fn(batch, attribute_padding_value):
        """
        return: tuple of tensors

        user: torch.tensor, shape [bs]
        target_item: torch.tensor, shape [bs]
        click_attributes: torch.tensor, shape [bs, max_click_num]
        nonclick_attributes: torch.tensor, shape [bs, max_nonclick_num]
        a_target: torch.tensor, shape [bs, max_a_target_num]
        cand_items: torch.tensor, shape [bs, max_cand_item_num]
        item_labels: torch.tensor, shape [bs, max_cand_item_num]
        click_attributes_len: torch.tensor, shape [bs]
        nonclick_attributes_len: torch.tensor, shape [bs]
        a_target_len: torch.tensor, shape [bs]

        """
        tuple_len = len(batch[0])
        user = torch.tensor([item[0] for item in batch], dtype=torch.int64)  
        target_item = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        click_attributes = pad_sequence([torch.tensor(item[2], dtype=torch.int64)  for item in batch], batch_first=True, padding_value=attribute_padding_value)
        nonclick_attributes = pad_sequence([torch.tensor(item[3], dtype=torch.int64)  for item in batch], batch_first=True, padding_value=attribute_padding_value)
        a_target = pad_sequence([torch.tensor(item[4], dtype=torch.int64) for item in batch], batch_first=True, padding_value=attribute_padding_value)
        
        click_attributes_len = torch.tensor([len(item[2]) for item in batch], dtype=torch.int64)
        nonclick_attributes_len = torch.tensor([len(item[3]) for item in batch], dtype=torch.int64)
        a_target_len = torch.tensor([len(item[4]) for item in batch], dtype=torch.int64)
        
        if tuple_len == 5:
            return user, target_item, click_attributes, nonclick_attributes, a_target, click_attributes_len, nonclick_attributes_len, a_target_len
        elif tuple_len >5:
            cand_items = torch.tensor([item[5] for item in batch], dtype=torch.int64)
            item_labels = torch.tensor([item[6] for item in batch], dtype=torch.int64)
            return user, target_item, click_attributes, nonclick_attributes, a_target, cand_items, item_labels, click_attributes_len, nonclick_attributes_len, a_target_len


def evaluate(model, test_dataloader, epoch, args):
    # === Test ===  
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        hit_at_k, ndcg_at_k = 0, 0
        precision, recall, f1 = 0, 0, 0
        print("Start testing {} user tuples".format(len(test_dataloader.dataset)))
        for i, batch in enumerate(tqdm(test_dataloader)):
            if i >= args.test_batch_count:
                print("Only test {} batches".format(args.test_batch_count))
                break
            user, target_item, click_attributes, nonclick_attributes, a_target, cand_items, item_labels, click_attributes_len, nonclick_attributes_len, a_target_len = batch
            user, target_item, click_attributes, nonclick_attributes, a_target, cand_items, item_labels, click_attributes_len, nonclick_attributes_len, a_target_len = user.to(args.device), target_item.to(args.device), click_attributes.to(args.device), nonclick_attributes.to(args.device), a_target.to(args.device), cand_items.to(args.device), item_labels.to(args.device), click_attributes_len.to(args.device), nonclick_attributes_len.to(args.device), a_target_len.to(args.device)
            click_attributes_len, nonclick_attributes_len, a_target_len = click_attributes_len.to(args.device), nonclick_attributes_len.to(args.device), a_target_len.to(args.device)

            pred_item, pred_attribute = model(user, click_attributes, nonclick_attributes, cand_items, click_attributes_len, nonclick_attributes_len)
            
            hit_at_k += calculate_hit_at_k(pred_item, (item_labels == 1).nonzero(as_tuple=True)[1], args.topk)
            ndcg_at_k += calculate_ndcg_at_k(pred_item, (item_labels == 1).nonzero(as_tuple=True)[1], args.topk)
            p, r, f = calculate_precision_recall_f1(pred_attribute, a_target)
            precision += p
            recall += r
            f1 += f

        hit_at_k /= args.test_batch_size*args.test_batch_count
        ndcg_at_k /= args.test_batch_size*args.test_batch_count
        precision /= args.test_batch_size*args.test_batch_count
        recall /= args.test_batch_size*args.test_batch_count
        f1 /= args.test_batch_size*args.test_batch_count
        print('epoch: {}, hit_at_k: {}, ndcg_at_k: {}, precision: {}, recall: {}, f1: {}'.format(epoch, hit_at_k, ndcg_at_k, precision, recall, f1))
        print('epoch: {}, time: {}'.format(epoch, time.time()-start_time))
        # logging.info('epoch: {}, hit_at_k: {}, ndcg_at_k: {}, precision: {}, recall: {}, f1: {}, time: {}'.format(epoch, hit_at_k, ndcg_at_k, precision, recall, f1, time.time()-start_time))



def main(args):
    user_item_dict, item_attribute_dict = load_simulaotr_dict_data(data_name, mode='train')
    test_user_item_dict, _ = load_simulaotr_dict_data(data_name, mode='test')
    # Parameters
   

    # create dataset and dataloader 
    train_dataset = UserSimulatorDataset(args.data_name, user_item_dict, item_attribute_dict, args.max_interactions, args.ask_num, args.alpha, args.change,  args.neg_sample_num, mode='train', num_workers=args.num_workers)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, attribute_count))

    test_dataset = UserSimulatorDataset(args.data_name, test_user_item_dict, item_attribute_dict, args.max_interactions, args.ask_num, args.alpha, args.change, args.test_neg_sample_num, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch, attribute_count))

    # Load retrieval graph
    retrieval_graph = load_graph(args.data_name)
    retrieval_graph.print_graph_info()
    
    
    user_count, item_count, attribute_count = retrieval_graph.node_counts['user'], retrieval_graph.node_counts['item'], retrieval_graph.node_counts['attribute']
    user_id_max, item_id_max, attribute_id_max = max(retrieval_graph.node_ids_by_type['user']), max(retrieval_graph.node_ids_by_type['item']), max(retrieval_graph.node_ids_by_type['attribute'])
    print('user_count: {}, item_count: {}, attribute_count: {}'.format(user_count, item_count, attribute_count))
    print('user_id_max: {}, item_id_max: {}, attribute_id_max: {}'.format(user_id_max, item_id_max, attribute_id_max))
    

    # Model
    model = EvolvingPreferenceModeling(user_id_max+1, item_id_max+1, attribute_id_max+1, args.hidden_size, args.seq_model, device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_attribute = torch.nn.BCEWithLogitsLoss()  # Masking for BCEWithLogitsLoss
    criterion_item = torch.nn.BCEWithLogitsLoss() 

    model.to(args.device)
    criterion_attribute.to(args.device)
    criterion_item.to(args.device)


    if args.load_model:
        print("Start Loading Model at epoch {}".format(args.load_model))
        save_path = TMP_DIR[args.data_name] + '/ranking_model/'
        model.load_state_dict(torch.load(save_path + 'system_ranking_model_{}.pt'.format(args.load_model)))
        print('model loaded from {}'.format(save_path + 'system_ranking_model_{}.pt'.format(args.load_model)))
        
        # === Generate Interaction ===
    
        test_dataloader.dataset.generate_interaction(load_from_file=False)
        evaluate(model, test_dataloader, epoch=-1, args=args)

        return 




    # Logging
    # === write logging ==
    logging.basicConfig(filename=f'{args.mv}[{args.data_name}].log', format = '%(asctime)s - %(name)s - %(message)s', datefmt="%d-%m-%y %H:%M:%S", level=logging.INFO)
    logging.info(f"================pid: {os.getpid()}================")
    args_dict = vars(args)
     
    logging.info('===== %s Namespace =====', args.data_name)
    log_line = ' '.join(['%s=%s | ' % (k, v) for k, v in args_dict.items()])
    logging.info(log_line)
    logging.info('=============================')
    
        
    best_hit = 0
    best_f1 = 0 
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        # === Generate Interaction ===
      
        if args.load_sample_data is False:
            if epoch % args.eopch_generate_interaction == 0:
                logging.info('epoch: {}, generate_interaction'.format(epoch))
                train_dataloader.dataset.generate_interaction(load_from_file=args.load_sample_data)
        else:
            load_epoch = epoch % args.eopch_generate_interaction
            logging.info('epoch: {}, load_interaction_epoch: {}'.format(epoch, load_epoch))
            train_dataloader.dataset.generate_interaction(load_from_file=args.load_sample_data, load_epoch=load_epoch)
        # === Train ===
        item_loss, attribute_loss = 0, 0
        for i, batch in enumerate(tqdm(train_dataloader)):
            user, target_item, click_attributes, nonclick_attributes, a_target, cand_items, item_labels, click_attributes_len, nonclick_attributes_len, a_target_len = batch
            user, target_item, click_attributes, nonclick_attributes, a_target, cand_items, item_labels, click_attributes_len, nonclick_attributes_len, a_target_len = user.to(args.device), target_item.to(args.device), click_attributes.to(args.device), nonclick_attributes.to(args.device), a_target.to(args.device), cand_items.to(args.device), item_labels.to(args.device), click_attributes_len.to(args.device), nonclick_attributes_len.to(args.device), a_target_len.to(args.device)
            click_attributes_len, nonclick_attributes_len, a_target_len = click_attributes_len.to(args.device), nonclick_attributes_len.to(args.device), a_target_len.to(args.device)

            def labels_to_binary_matrix(labels, num_labels, padding_value=-1):
                # Create a binary matrix of size (batch_size, num_labels) 
                binary_matrix = torch.zeros(labels.size(0), num_labels, dtype=torch.float32).to(labels.device)

                # Set the indices of the labels that are not equal to the padding value to 1
                mask = labels != padding_value
                indices = labels[mask]
                row_indices = torch.arange(labels.size(0)).unsqueeze(-1).expand_as(labels).to(labels.device)
                row_indices = row_indices[mask]
                binary_matrix[row_indices, indices] = 1

                return binary_matrix

            attribute_labels = labels_to_binary_matrix(a_target, attribute_count, padding_value=attribute_count)


            optimizer.zero_grad()
            # user_ids, attribute_click_sequence, attribute_nonclick_sequence, cand_items, seq_click_lengths, seq_nonclick_lengths
            
            pred_item, pred_attribute = model(user, click_attributes, nonclick_attributes, cand_items, click_attributes_len, nonclick_attributes_len)
            loss_item = criterion_item(pred_item, item_labels.float())
            loss_attribute = criterion_attribute(pred_attribute,attribute_labels.float())

            
            loss = args.alpha_1 * loss_item + args.alpha_2 * loss_attribute
            loss.backward()
            optimizer.step()
            
            
            item_loss += loss_item.item()
            attribute_loss += loss_attribute.item()

            
        print('epoch: {}, item_loss: {}, attribute_loss: {}'.format(epoch, item_loss, attribute_loss))
        print('epoch: {}, time: {}'.format(epoch, time.time()-start_time))
        logging.info('epoch: {}, item_loss: {}, attribute_loss: {}, time: {}'.format(epoch, item_loss, attribute_loss, time.time()-start_time))
        
         
        # === Test ===  
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            # === Generate Interaction ===
            if args.load_sample_data is False:
                if epoch % args.eopch_generate_interaction == 0:
                    logging.info('epoch: {}, generate_test_interaction'.format(epoch))
                    test_dataloader.dataset.generate_interaction(load_from_file=args.load_sample_data)
            else:
                load_epoch = epoch % args.eopch_generate_interaction
                logging.info('epoch: {}, load_test_interaction_epoch: {}'.format(epoch, load_epoch))
                test_dataloader.dataset.generate_interaction(load_from_file=args.load_sample_data, load_epoch=load_epoch)
            
            hit_at_k, ndcg_at_k = 0, 0
            precision, recall, f1 = 0, 0, 0
            print("Start testing {} user tuples".format(len(test_dataloader.dataset)))
            for i, batch in enumerate(tqdm(test_dataloader)):
                if i >= args.test_batch_count:
                    print("Only test {} batches".format(args.test_batch_count))
                    break
                user, target_item, click_attributes, nonclick_attributes, a_target, cand_items, item_labels, click_attributes_len, nonclick_attributes_len, a_target_len = batch
                user, target_item, click_attributes, nonclick_attributes, a_target, cand_items, item_labels, click_attributes_len, nonclick_attributes_len, a_target_len = user.to(args.device), target_item.to(args.device), click_attributes.to(args.device), nonclick_attributes.to(args.device), a_target.to(args.device), cand_items.to(args.device), item_labels.to(args.device), click_attributes_len.to(args.device), nonclick_attributes_len.to(args.device), a_target_len.to(args.device)
                click_attributes_len, nonclick_attributes_len, a_target_len = click_attributes_len.to(args.device), nonclick_attributes_len.to(args.device), a_target_len.to(args.device)

                pred_item, pred_attribute = model(user, click_attributes, nonclick_attributes, cand_items, click_attributes_len, nonclick_attributes_len)
                
                hit_at_k += calculate_hit_at_k(pred_item, (item_labels == 1).nonzero(as_tuple=True)[1], args.topk)
                ndcg_at_k += calculate_ndcg_at_k(pred_item, (item_labels == 1).nonzero(as_tuple=True)[1], args.topk)
                p, r, f = calculate_precision_recall_f1(pred_attribute, a_target)
                precision += p
                recall += r
                f1 += f

            hit_at_k /= args.test_batch_size*args.test_batch_count
            ndcg_at_k /= args.test_batch_size*args.test_batch_count
            precision /= args.test_batch_size*args.test_batch_count
            recall /= args.test_batch_size*args.test_batch_count
            f1 /= args.test_batch_size*args.test_batch_count
            print('epoch: {}, hit_at_k: {}, ndcg_at_k: {}, precision: {}, recall: {}, f1: {}'.format(epoch, hit_at_k, ndcg_at_k, precision, recall, f1))
            print('epoch: {}, time: {}'.format(epoch, time.time()-start_time))
            logging.info('epoch: {}, hit_at_k: {}, ndcg_at_k: {}, precision: {}, recall: {}, f1: {}, time: {}'.format(epoch, hit_at_k, ndcg_at_k, precision, recall, f1, time.time()-start_time))

        # === Save Model ===
        if hit_at_k > best_hit and f1 > best_f1:
            best_hit = hit_at_k
            best_f1 = f1
        
            save_path = TMP_DIR[args.data_name] + '/ranking_model/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + f'{args.mv}_model.pt')
            print('model saved to {}'.format(save_path + f'{args.mv}_model.pt'))
            logging.info('Saving model successfully!')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=YELP_STAR, choices=[LAST_FM_STAR, YELP_STAR, BOOK],
                        help='One of { LAST_FM_STAR, YELP_STAR, BOOK}.')
    parser.add_argument('--mv', type=str, default='user_model_v1', help='model_version')

    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='device')
    parser.add_argument('--gpu', type=str, default='6', help='gpu device.')
    parser.add_argument('--load_model', type=int, default=0, help='load_model') #TODO

    
    # Sample parameters 
    parser.add_argument('--max_interactions', type=int, default=10, help='max_interactions')
    parser.add_argument('--ask_num', type=int, default=2, help='ask_num')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--change', type=float, default=0.1, help='change')
    parser.add_argument('--neg_sample_num', type=int, default=2000, help='negtive sample items number')
    parser.add_argument('--test_neg_sample_num', type=int, default=2000, help='negtive sample items number')
    parser.add_argument('--num_workers', type=int, default=64, help='num_workers')

    
    # Model parameters
    parser.add_argument('--epochs', type=int, default=300, help='epochs')
    parser.add_argument('--load_sample_data', type=bool, default=False, help='load_saved_data')
    parser.add_argument('--eopch_generate_interaction', type=int, default=10, help='generate_sample_data when load_sample_data is False')
    parser.add_argument('--train_batch_size', type=int, default=2048, help='train_batch_size')
    parser.add_argument('--test_batch_size', type=int, default=512, help='test_batch_size')
    parser.add_argument('--test_batch_count', type=int, default=5, help='test_batch_num')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden_size')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--seq_model', type=str, default='transformer', choices=['gru', 'lstm', 'transformer'], help='seq_model')

    # Multi-Loss parameters
    parser.add_argument('--alpha_1', type=float, default=0.8, help='item_loss_weight')
    parser.add_argument('--alpha_2', type=float, default=0.2, help='attribute_loss_weight')

    #Evaluation parameters
    parser.add_argument('--topk', type=int, default=10, help='topk items for evaluation')



    args = parser.parse_args()
    data_name = args.data_name
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    set_random_seed(2024)

    main(args)

    