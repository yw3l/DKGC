import torch
import torch.nn as nn
import torch.nn.functional as F

class DKGCModel(nn.Module):
    def __init__(self, structural_encoder, textual_encoder, 
                 struct_dim=100, text_dim=768, projection_dim=256, gamma=12.0):
        super(DKGCModel, self).__init__()
        self.structural_encoder = structural_encoder
        self.textual_encoder = textual_encoder
        self.gamma = gamma

        # Projection matrices
        self.W_s = nn.Linear(struct_dim, projection_dim)
        self.W_t = nn.Linear(text_dim, projection_dim)

        # Gating MLP
        # Input: [v_struct; v_text] -> dim: 2 * projection_dim + 2 * projection_dim = 1024
        self.gating_mlp = nn.Sequential(
            nn.Linear(4 * projection_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Fusion MLP for nonlinear interaction
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * projection_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, struct_sample, text_inputs):
        """
        struct_sample: tensor of (h_id, r_id, t_id)
        text_inputs: dict of tokenized inputs for SimKGC
        """
        # 1. Extract raw features
        # h_prime_s, t_s: (batch_size, struct_dim)
        h_prime_s, t_s = self.structural_encoder.get_structural_embeddings(struct_sample)
        
        # hr_text, t_text: (batch_size, text_dim)
        text_outputs = self.textual_encoder(**text_inputs)
        hr_text = text_outputs['hr_vector']
        t_text = text_outputs['tail_vector']

        # 2. Projection
        h_prime_s_tilde = self.W_s(h_prime_s)
        t_s_tilde = self.W_s(t_s)
        hr_text_tilde = self.W_t(hr_text)
        t_text_tilde = self.W_t(t_text)

        # 3. Construct interaction features
        v_struct = torch.cat([h_prime_s_tilde, t_s_tilde], dim=-1) # (batch, 2*projection_dim)
        v_text = torch.cat([hr_text_tilde, t_text_tilde], dim=-1)     # (batch, 2*projection_dim)
        v_combined = torch.cat([v_struct, v_text], dim=-1)         # (batch, 4*projection_dim)

        # 4. Gating
        alpha = self.gating_mlp(v_combined) # (batch, 1)

        # 5. Scores
        # Textual score: dot product with temperature
        s_text = torch.sum(hr_text * t_text, dim=-1, keepdim=True)
        if hasattr(self.textual_encoder, 'log_inv_t'):
            s_text = s_text * self.textual_encoder.log_inv_t.exp()
        
        # Structural score in projected space
        s_struct_prime = self.gamma - torch.norm(h_prime_s_tilde - t_s_tilde, p=1, dim=-1, keepdim=True)

        s_gate = alpha * s_text + (1 - alpha) * s_struct_prime

        # 6. Nonlinear interaction
        f_fusion = self.fusion_mlp(v_combined)

        # 7. Final score
        s_final = s_gate + f_fusion

        return s_final.squeeze(-1)
