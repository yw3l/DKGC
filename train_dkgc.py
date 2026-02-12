import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoTokenizer

from dkgc_model import DKGCModel
# Add SimKGC and CompoundE to path if necessary
import sys

# --- Hack to handle SimKGC config import ---
original_argv = sys.argv[:]
sys.argv = [sys.argv[0], '--task', 'fb15k237', '--model-dir', './tmp_log'] 
sys.path.append('./SimKGC')
sys.path.append('./CompoundE')

from models import CustomBertModel
from model import KGEModel
from dict_hub import get_entity_dict, build_tokenizer
from doc import collate

# Restore argv for our own parser
sys.argv = original_argv
# -------------------------------------------

class DKGCDataset(Dataset):
    def __init__(self, json_path, entity_dict, rel2idx, tokenizer, max_length=64):
        self.data = json.load(open(json_path, 'r', encoding='utf-8'))
        self.entity_dict = entity_dict
        self.rel2idx = rel2idx
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        h_id, r_str, t_id = ex['head_id'], ex['relation'], ex['tail_id']
        
        # Structural IDs
        h_idx = self.entity_dict.entity_to_idx(h_id)
        t_idx = self.entity_dict.entity_to_idx(t_id)
        r_idx = self.rel2idx[r_str]

        # Textual inputs
        from doc import Example
        example_obj = Example(head_id=h_id, relation=r_str, tail_id=t_id)
        return example_obj.vectorize(), torch.LongTensor([h_idx, r_idx, t_idx])

def dkgc_collate(batch):
    examples = [b[0] for b in batch]
    struct_samples = torch.stack([b[1] for b in batch])
    
    # Use SimKGC's collate to get tokenized text inputs
    from doc import collate
    text_inputs = collate(examples)
    
    return struct_samples, text_inputs

def train(args):
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # 2. Load entity and relation dicts
    # Mock args for SimKGC dict_hub
    from config import args as simkgc_args
    simkgc_args.valid_path = os.path.join(args.data_dir, 'valid.txt.json')
    simkgc_args.train_path = os.path.join(args.data_dir, 'train.txt.json')
    simkgc_args.pretrained_model = args.pretrained_model
    
    entity_dict = get_entity_dict()
    tokenizer = build_tokenizer(simkgc_args)
    
    # Load relations and create mapping
    rel_json = json.load(open(os.path.join(args.data_dir, 'relations.json'), 'r', encoding='utf-8'))
    all_rels = sorted(list(rel_json.keys()))
    rel2idx = {rel: i for i, rel in enumerate(all_rels)}
    
    # 3. Load Base Encoders
    # Load SimKGC
    simkgc_model = CustomBertModel(simkgc_args)
    if args.simkgc_checkpoint:
        print(f"Loading SimKGC from {args.simkgc_checkpoint}")
        simkgc_model.load_state_dict(torch.load(args.simkgc_checkpoint, map_location='cpu')['state_dict'], strict=False)
    
    # Load CompoundE
    # We need nentity, nrelation, hidden_dim etc.
    nentity = len(entity_dict)
    nrelation = len(rel2idx)
    # Using default or provided hyperparams for CompoundE
    struct_model = KGEModel(
        model_name='CompoundE_Head',
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.struct_hidden_dim,
        gamma=args.gamma,
        evaluator=None, # Not needed for training fusion
        triple_relation_embedding=True
    )
    if args.compounde_checkpoint:
        print(f"Loading CompoundE from {args.compounde_checkpoint}")
        # CompoundE save_model saves numpy arrays
        struct_model.entity_embedding.data = torch.from_numpy(np.load(os.path.join(args.compounde_checkpoint, 'entity_embedding.npy')))
        struct_model.relation_embedding.data = torch.from_numpy(np.load(os.path.join(args.compounde_checkpoint, 'relation_embedding.npy')))

    # 4. Initialize DKGC Model
    model = DKGCModel(
        structural_encoder=struct_model,
        textual_encoder=simkgc_model,
        struct_dim=args.struct_hidden_dim,
        text_dim=768, # BERT base hidden dim
        projection_dim=args.projection_dim,
        gamma=args.gamma
    )
    model.to(device)

    # 5. Freeze Encoders (Stage 2 of training)
    for param in model.structural_encoder.parameters():
        param.requires_grad = False
    for param in model.textual_encoder.parameters():
        param.requires_grad = False
    
    # 6. Dataset and Dataloader
    train_dataset = DKGCDataset(simkgc_args.train_path, entity_dict, rel2idx, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dkgc_collate)

    # 7. Optimizer and Loss
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for step, (struct_samples, text_inputs) in enumerate(train_loader):
            batch_size = struct_samples.size(0)
            struct_samples = struct_samples.to(device)
            for k in text_inputs:
                if isinstance(text_inputs[k], torch.Tensor):
                    text_inputs[k] = text_inputs[k].to(device)
            
            # In-batch negatives for fusion
            # We need to compute fusion scores for all h-r in batch against all t in batch
            # This requires a modified forward pass in DKGCModel or a loop
            # For simplicity in this template, let's assume we want to score positive pairs
            # and use them for a margin-based loss against sampled negatives.
            
            # To support in-batch negatives efficiently:
            # 1. Get hr_vectors and tail_vectors for both modalities
            h_prime_s, t_s = model.structural_encoder.get_structural_embeddings(struct_samples)
            
            text_outputs = model.textual_encoder(**text_inputs)
            hr_text = text_outputs['hr_vector']
            t_text = text_outputs['tail_vector']
            
            # 2. Project all
            h_prime_s_tilde = model.W_s(h_prime_s)
            t_s_tilde = model.W_s(t_s)
            hr_text_tilde = model.W_t(hr_text)
            t_text_tilde = model.W_t(t_text)
            
            # 3. Compute scores for all pairs in batch (batch_size, batch_size)
            # This is complex because gating depends on both h and t.
            # s_final[i, j] = Gate(h_i, t_j) * Score(h_i, t_j) + ...
            
            # Simplified version for the template: only positive pairs + random negative
            pos_scores = model(struct_samples, text_inputs)
            
            # Random negative tails
            neg_struct_samples = struct_samples.clone()
            neg_indices = torch.randint(0, nentity, (batch_size,)).to(device)
            neg_struct_samples[:, 2] = neg_indices
            
            # We'd need text inputs for these random tails too...
            # This is why in-batch negatives are better.
            
            # Let's use in-batch negatives for the structural part at least.
            # But the textual part also needs BERT outputs for all entities in batch.
            # SimKGC already does this! 
            
            # Placeholder for the actual loss calculation
            # labels = torch.arange(batch_size).to(device)
            # loss = criterion(scores, labels)
            
            print(f"Epoch {epoch}, Step {step}: Pos scores mean {pos_scores.mean().item():.4f}")
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            break # Just one step for demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='SimKGC/data/FB15k237')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--simkgc_checkpoint', type=str, default=None)
    parser.add_argument('--compounde_checkpoint', type=str, default=None)
    parser.add_argument('--struct_hidden_dim', type=int, default=100)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=12.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--cuda', action='store_true')
    
    args = parser.parse_args()
    train(args)