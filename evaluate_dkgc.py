import os
import json
import torch
import argparse
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# --- Hack to handle SimKGC config import ---
# SimKGC's config.py parses args immediately on import. 
# We need to temporarily set sys.argv to avoid errors with unknown args from evaluate_dkgc.
original_argv = sys.argv[:]
sys.argv = [sys.argv[0], '--task', 'fb15k237', '--model-dir', './tmp_log'] # Minimal valid args for SimKGC
sys.path.append('./SimKGC')
sys.path.append('./CompoundE')

from dkgc_model import DKGCModel
from models import CustomBertModel
from model import KGEModel
from dict_hub import get_entity_dict, build_tokenizer, get_tokenizer
from doc import collate, Example
import config # Import config to access/modify args later

# Restore argv for our own parser
sys.argv = original_argv
# -------------------------------------------

class TestDataset(Dataset):
    def __init__(self, json_path, entity_dict, rel2idx):
        self.data = json.load(open(json_path, 'r', encoding='utf-8'))
        self.entity_dict = entity_dict
        self.rel2idx = rel2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        h_id, r_str, t_id = ex['head_id'], ex['relation'], ex['tail_id']
        h_idx = self.entity_dict.entity_to_idx(h_id)
        t_idx = self.entity_dict.entity_to_idx(t_id)
        
        # Handle potential unknown relation in test set? 
        # For FB15k237 it should be fine.
        if r_str in self.rel2idx:
            r_idx = self.rel2idx[r_str]
        else:
            # Fallback or error? Let's use 0 and warn if strict
            r_idx = 0 
        
        # Create Example object and vectorize
        example_obj = Example(head_id=h_id, head=ex['head'], relation=r_str, tail_id=t_id, tail=ex['tail'])
        vectorized_ex = example_obj.vectorize()
        
        return vectorized_ex, torch.LongTensor([h_idx, r_idx, t_idx])

def eval_collate(batch):
    # batch is list of tuples (vectorized_dict, struct_tensor)
    vectorized_batch = [b[0] for b in batch]
    struct_samples = torch.stack([b[1] for b in batch])
    return vectorized_batch, struct_samples

@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    # --- 1. Load Resources ---
    from config import args as simkgc_args
    simkgc_args.pretrained_model = args.pretrained_model
    # Mock data paths for dict_hub
    simkgc_args.train_path = os.path.join(args.data_dir, 'train.txt.json')
    simkgc_args.valid_path = os.path.join(args.data_dir, 'valid.txt.json')
    
    entity_dict = get_entity_dict()
    build_tokenizer(simkgc_args)
    tokenizer = get_tokenizer()
    
    rel_json = json.load(open(os.path.join(args.data_dir, 'relations.json'), 'r', encoding='utf-8'))
    all_rels = sorted(list(rel_json.keys()))
    rel2idx = {rel: i for i, rel in enumerate(all_rels)}
    
    # --- 2. Load Models ---
    print("Loading models...")
    # SimKGC
    simkgc_model = CustomBertModel(simkgc_args)
    
    # CompoundE
    struct_model = KGEModel(
        model_name='CompoundE_Head',
        nentity=len(entity_dict),
        nrelation=len(rel2idx),
        hidden_dim=args.struct_hidden_dim,
        gamma=args.gamma,
        evaluator=None,
        triple_relation_embedding=True
    )
    
    # DKGC Wrapper
    model = DKGCModel(
        structural_encoder=struct_model,
        textual_encoder=simkgc_model,
        struct_dim=args.struct_hidden_dim,
        text_dim=768,
        projection_dim=args.projection_dim,
        gamma=args.gamma
    )
    
    if args.dkgc_checkpoint:
        print(f"Loading DKGC checkpoint from {args.dkgc_checkpoint}")
        checkpoint = torch.load(args.dkgc_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    
    model.to(device)
    model.eval()
    
    # --- 3. Pre-compute Entity Embeddings ---
    print("Pre-computing entity embeddings...")
    # 3.1 Structural Embeddings (t_s)
    # These are just the entity embeddings from CompoundE
    all_struct_ent_embs = model.structural_encoder.entity_embedding.data.to(device) # (N, struct_dim)
    
    # 3.2 Textual Embeddings (t_text)
    # We need to run BERT on all entity descriptions
    all_text_ent_embs = []
    # Batch process entities
    ent_batch_size = args.batch_size * 2
    all_entities = [entity_dict.get_entity_by_idx(i) for i in range(len(entity_dict))]
    
    for i in tqdm(range(0, len(all_entities), ent_batch_size), desc="Encoding entities"):
        batch_ents = all_entities[i : i + ent_batch_size]
        # Tokenize
        batch_text = [ex.entity + ' ' + ex.entity_desc for ex in batch_ents]
        # SimKGC uses 'tail_bert' for entities. 
        # We use 'predict_ent_embedding' from CustomBertModel which handles tokenization internally?
        # No, CustomBertModel expects token_ids.
        
        features = tokenizer(batch_text, max_length=args.max_length, truncation=True, padding=True, return_tensors='pt')
        features = {k: v.to(device) for k, v in features.items()}
        
        # SimKGC model signature for ent embedding:
        # predict_ent_embedding(tail_token_ids, tail_mask, tail_token_type_ids)
        outputs = model.textual_encoder.predict_ent_embedding(
            tail_token_ids=features['input_ids'],
            tail_mask=features['attention_mask'],
            tail_token_type_ids=features.get('token_type_ids')
        )
        all_text_ent_embs.append(outputs['ent_vectors'])
        
    all_text_ent_embs = torch.cat(all_text_ent_embs, dim=0) # (N, 768)
    
    # Pre-project embeddings to speed up fusion?
    # t_s_tilde = W_s(t_s)
    # t_text_tilde = W_t(t_text)
    # This optimization allows us to skip projection during re-ranking
    print("Projecting entity embeddings...")
    all_struct_ent_proj = model.W_s(all_struct_ent_embs) # (N, 256)
    all_text_ent_proj = model.W_t(all_text_ent_embs)     # (N, 256)
    
    # --- 4. Evaluation Loop ---
    test_path = os.path.join(args.data_dir, 'test.txt.json')
    test_dataset = TestDataset(test_path, entity_dict, rel2idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_collate)
    
    logs = []
    k = args.rerank_k
    
    print(f"Starting evaluation on {len(test_dataset)} triples...")
    for step, (vectorized_batch, struct_samples) in enumerate(tqdm(test_loader, desc="Eval")):
        batch_size = len(vectorized_batch)
        struct_samples = struct_samples.to(device) # (B, 3)
        
        # 4.1 Phase 1: Retrieval (SimKGC)
        # Encode queries (h, r)
        # SimKGC expects hr_token_ids etc. from doc.collate
        batch_inputs = collate(vectorized_batch)
        # We only need HR part inputs
        hr_inputs = {
            'hr_token_ids': batch_inputs['hr_token_ids'].to(device),
            'hr_mask': batch_inputs['hr_mask'].to(device),
            'hr_token_type_ids': batch_inputs['hr_token_type_ids'].to(device)
        }
        
        # Get query text embeddings
        # model.textual_encoder forward returns dict with 'hr_vector'
        # But we need to call partial forward or access internal BERT
        # CustomBertModel doesn't expose a clean "encode_query" method, but forward does it.
        # We can pass dummy tail inputs to satisfy forward, or just use _encode directly if accessible.
        # Or better: construct full inputs with dummy tails and ignore tail outputs.
        # Actually, CustomBertModel forward computes hr_vector if inputs provided.
        # We can pass None for tail inputs? No, forward signature requires them.
        # Let's verify CustomBertModel again.
        
        # It calls self._encode(self.hr_bert, ...)
        # We can manually call _encode
        hr_vector = model.textual_encoder._encode(
            model.textual_encoder.hr_bert,
            token_ids=hr_inputs['hr_token_ids'],
            mask=hr_inputs['hr_mask'],
            token_type_ids=hr_inputs['hr_token_type_ids']
        ) # (B, 768)
        
        # Retrieval Scores (Dot product)
        # (B, 768) @ (N, 768).T -> (B, N)
        simkgc_scores = torch.mm(hr_vector, all_text_ent_embs.t())
        
        # Get Top-K candidates
        # We should include the ground truth if it's not in top-K to compute accurate metrics?
        # Standard MRR: rank among ALL entities.
        # DKGC Strategy: Re-rank Top-K. If GT not in Top-K, rank is K+something?
        # Usually for "Rescoring" methods, we assume if GT is missed, it's a failure.
        # But to be fair, let's just re-rank Top-K.
        topk_scores, topk_indices = torch.topk(simkgc_scores, k=k, dim=1) # (B, K)
        
        # 4.2 Phase 2: Re-ranking (Fusion)
        # We need to compute DKGC scores for these K candidates.
        
        # Prepare Query Features
        # Structural: h_prime_s
        # We need to compute h_prime_s for the batch queries
        h_prime_s, _ = model.structural_encoder.get_structural_embeddings(struct_samples) # (B, struct_dim)
        h_prime_s_tilde = model.W_s(h_prime_s) # (B, 256)
        
        # Textual: hr_vector (already computed)
        hr_vector_tilde = model.W_t(hr_vector) # (B, 256)
        
        # Expand queries to match K candidates: (B, 1, 256) -> (B, K, 256)
        h_prime_s_exp = h_prime_s_tilde.unsqueeze(1).expand(-1, k, -1)
        hr_vector_exp = hr_vector_tilde.unsqueeze(1).expand(-1, k, -1)
        
        # Prepare Candidate Features
        # Indices: topk_indices (B, K)
        flat_indices = topk_indices.view(-1)
        
        # Structural Candidates (Projected)
        cand_struct_tilde = torch.index_select(all_struct_ent_proj, 0, flat_indices).view(batch_size, k, -1)
        
        # Textual Candidates (Projected)
        cand_text_tilde = torch.index_select(all_text_ent_proj, 0, flat_indices).view(batch_size, k, -1)
        
        # Construct Interaction Features
        # v_struct: [h'; t]
        v_struct = torch.cat([h_prime_s_exp, cand_struct_tilde], dim=-1) # (B, K, 512)
        
        # v_text: [hr; t]
        v_text = torch.cat([hr_vector_exp, cand_text_tilde], dim=-1) # (B, K, 512)
        
        v_combined = torch.cat([v_struct, v_text], dim=-1) # (B, K, 1024)
        
        # Run Fusion MLPs
        # Gating
        alpha = model.gating_mlp(v_combined) # (B, K, 1)
        
        # Scores
        # S_text: we can use the topk_scores (dot product) we already have!
        # topk_scores is (B, K)
        s_text = topk_scores.unsqueeze(-1)
        
        # S_struct: L1 distance in projected space
        # dist = || h' - t ||
        dist_struct = torch.norm(h_prime_s_exp - cand_struct_tilde, p=1, dim=-1, keepdim=True)
        s_struct_prime = model.gamma - dist_struct
        
        s_gate = alpha * s_text + (1 - alpha) * s_struct_prime
        
        # Nonlinear Fusion
        f_fusion = model.fusion_mlp(v_combined) # (B, K, 1)
        
        s_final = s_gate + f_fusion # (B, K, 1)
        s_final = s_final.squeeze(-1) # (B, K)
        
        # 4.3 Compute Metrics
        # Check where the ground truth is
        # GT indices: struct_samples[:, 2] (B,)
        gt_indices = struct_samples[:, 2].unsqueeze(1) # (B, 1)
        
        # Check if GT is in Top-K
        # topk_indices: (B, K)
        is_hit = (topk_indices == gt_indices) # (B, K) boolean
        
        # We need to find the rank in the re-ranked list
        # Sort s_final descending
        sorted_scores, sorted_indices_local = torch.sort(s_final, descending=True, dim=1)
        
        # Map local indices back to global entity indices?
        # Actually we just need to know where the hit went.
        # But we also need to handle the case where GT was NOT in retrieval top-K.
        
        batch_ranks = []
        for i in range(batch_size):
            # Check if GT was in candidates
            row_hit = is_hit[i] # (K,)
            if row_hit.any():
                # GT was retrieved. Find its new rank.
                # row_hit has exactly one True.
                # Find which local index corresponds to GT
                local_gt_idx = row_hit.nonzero(as_tuple=True)[0].item()
                
                # Where is this local_idx in sorted_indices_local[i]?
                # sorted_indices_local[i] contains permutations of 0..K-1
                rank_in_candidates = (sorted_indices_local[i] == local_gt_idx).nonzero(as_tuple=True)[0].item() + 1
                batch_ranks.append(rank_in_candidates)
            else:
                # GT not retrieved. Rank is > K.
                # Use SimKGC rank or just penalty?
                # For strict Re-ranking metrics, we often assume Rank = K + 1 or infinite.
                # Let's assume SimKGC rank was > K.
                # To be precise, we should probably check SimKGC full rank, but that's expensive.
                # A common approximation is to set rank = K + 1 or just ignore (but that boosts scores).
                # Setting to K+1 is a "optimistic lower bound" for the error. 
                # Better: calculate SimKGC rank for this missed one?
                # No, let's stick to the behavior: "re-rank top-K".
                # If not in top-K, we can't improve it. 
                # We count it as rank > 10 (since we usually care about Hits@10).
                batch_ranks.append(10000) # Large number
        
        logs += batch_ranks

    # --- 5. Summary ---
    logs = np.array(logs)
    mrr = 1.0 / logs
    mrr = np.mean(mrr)
    
    hits1 = np.mean(logs <= 1)
    hits3 = np.mean(logs <= 3)
    hits10 = np.mean(logs <= 10)
    
    print(f"\nResults:")
    print(f"MRR: {mrr:.4f}")
    print(f"Hits@1: {hits1:.4f}")
    print(f"Hits@3: {hits3:.4f}")
    print(f"Hits@10: {hits10:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='SimKGC/data/FB15k237')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--dkgc_checkpoint', type=str, default=None)
    parser.add_argument('--struct_hidden_dim', type=int, default=100)
    parser.add_argument('--projection_dim', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=12.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--rerank_k', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--cuda', action='store_true')
    
    args = parser.parse_args()
    evaluate(args)
