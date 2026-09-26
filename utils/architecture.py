import torch
import torch.nn as nn
import torch.nn.functional as F

def build_branch(branch_type, input_dim, num_classes, R, device,
                 hyperparams, num_heads=4, pool_method="attn"):
    if branch_type == "unimodal":
        branch = SelfAttentionPoolingWithConstraint(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=num_classes,
            hyperparams=hyperparams,
            R=R,
            num_heads=num_heads,
        )
    elif branch_type == "cross":
        branch = CrossAttentionFusionWithConstraint(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=num_classes,
            hyperparams=hyperparams,
            R=R,
            num_heads=num_heads,
            pool_method=pool_method,
        )
    else:
        raise ValueError(f"Unknown branch_type: {branch_type}")

    branch.to(device)
    return branch

def build_branch_singlepath(branch_type, input_dim, num_classes, R, device,
                            hyperparams, leaf_index, num_heads=4, pool_method="attn"):
    if branch_type == "unimodal":
        branch = SelfAttentionPoolingSinglePath(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=num_classes,
            hyperparams=hyperparams,
            R=R,
            leaf_index=leaf_index,
            num_heads=num_heads,
        )
    elif branch_type == "cross":
        branch = CrossAttentionFusionSinglePath(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=num_classes,
            hyperparams=hyperparams,
            R=R,
            leaf_index=leaf_index,
            num_heads=num_heads,
            pool_method=pool_method,
        )
    else:
        raise ValueError(f"Unknown branch_type: {branch_type}")

    branch.to(device)
    return branch

def load_branch(checkpoint_path, branch_type, input_dim, num_classes, R, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt["hyperparams"]
    saved_args = ckpt.get("args", {})

    branch = build_branch(
        branch_type, input_dim, num_classes, R, device, hp,
        num_heads=saved_args.get("num_heads", 4),
        pool_method=saved_args.get("pool_method", "attn"),
    )

    branch.load_state_dict(ckpt["model_state_dict"])
    branch.to(device)
    return branch, hp

def load_branch_singlepath(checkpoint_path, branch_type, input_dim, num_classes, R, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt["hyperparams"]
    saved_args = ckpt.get("args", {})
    state = ckpt["model_state_dict"]

    leaf_index = state.get("constrained_model.leaf_index")
    if leaf_index is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} has no "
            f"'constrained_model.leaf_index'; expected a single-path branch."
        )

    if branch_type == "unimodal":
        branch = SelfAttentionPoolingSinglePath(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=num_classes,
            hyperparams=hp,
            R=R,
            leaf_index=leaf_index,
            num_heads=saved_args.get("num_heads", 4),
        )
    elif branch_type == "cross":
        branch = CrossAttentionFusionSinglePath(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=num_classes,
            hyperparams=hp,
            R=R,
            leaf_index=leaf_index,
            num_heads=saved_args.get("num_heads", 4),
            pool_method=saved_args.get("pool_method", "attn"),
        )
    else:
        raise ValueError(f"Unknown branch_type: {branch_type}")

    branch.load_state_dict(state)
    branch.to(device)
    return branch, hp

def get_constr_out(x, R):
    c_out = x
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out, dim=-1)
    return final_out

def softmax_constr_out(logits, R, leaf_index):
    B, C = logits.shape
    leaf_logits = logits.index_select(1, leaf_index)
    leaf_probs = F.softmax(leaf_logits, dim=-1)
    full = logits.new_zeros(B, C)
    full.index_copy_(1, leaf_index, leaf_probs)
    coherent = full @ R.t()
    return coherent

class ConstrainedFFNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(ConstrainedFFNNModel, self).__init__()

        self.nb_layers = hyperparams['num_layers']
        self.R = R

        fc = []
        for i in range(self.nb_layers):
            if self.nb_layers == 1:
                fc.append(nn.Linear(input_dim, output_dim))
            elif i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(hyperparams['dropout'])

        self.sigmoid = nn.Sigmoid()
        if hyperparams['activation'] == 'tanh':
            self.f = nn.Tanh()
        elif hyperparams['activation'] == 'relu':
            self.f = nn.ReLU()
        else:
            self.f = nn.GELU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.sigmoid(self.fc[i](x))
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            return x
        else:
            constrained_out = get_constr_out(x, self.R)
            return x, constrained_out

class SinglePathFFNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R, leaf_index):
        super().__init__()
        self.nb_layers = hyperparams['num_layers']
        self.R = R
        self.register_buffer(
            "leaf_index", torch.as_tensor(leaf_index, dtype=torch.long)
        )

        fc = []
        for i in range(self.nb_layers):
            if self.nb_layers == 1:
                fc.append(nn.Linear(input_dim, output_dim))
            elif i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(hyperparams['dropout'])
        if hyperparams['activation'] == 'tanh':
            self.f = nn.Tanh()
        elif hyperparams['activation'] == 'relu':
            self.f = nn.ReLU()
        else:
            self.f = nn.GELU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.fc[i](x)
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        if self.training:
            return x
        else:
            coherent = softmax_constr_out(x, self.R, self.leaf_index)
            return x, coherent


class GatedFusion(nn.Module):
    def __init__(self, emb_dim, init_bias=2.0):
        super().__init__()
        self.gate = nn.Linear(emb_dim * 2, emb_dim)
        nn.init.constant_(self.gate.bias, init_bias)

    def forward(self, img, txt):
        concat = torch.cat([img, txt], dim=-1)
        alpha = torch.sigmoid(self.gate(concat))
        fused = alpha * img + (1 - alpha) * txt
        return fused

class GatedFusionSinglePath(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R,
                 leaf_index, init_bias=2.0):
        super().__init__()
        self.fusion = GatedFusion(input_dim, init_bias)
        self.constrained_model = SinglePathFFNNModel(
            input_dim, hidden_dim, output_dim, hyperparams, R, leaf_index
        )

    def forward(self, img, txt):
        fused = self.fusion(img, txt)
        return self.constrained_model(fused)

class GatedFusionWithConstraint(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R, init_bias=2.0):
        super().__init__()
        self.fusion = GatedFusion(input_dim, init_bias)
        self.constrained_model = ConstrainedFFNNModel(
            input_dim, hidden_dim, output_dim, hyperparams, R
        )

    def forward(self, img, txt):
        fused = self.fusion(img, txt)
        return self.constrained_model(fused)


def _masked_mean(x, mask):
    if mask is None:
        return x.mean(dim=1)
    x = x * mask.unsqueeze(-1)
    lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return x.sum(dim=1) / lengths


def _to_key_padding_mask(mask):
    if mask is None:
        return None
    return ~mask


class CrossAttentionFusion(nn.Module):
    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, q_mask=None, kv_mask=None):
        kv_key_pad_mask = _to_key_padding_mask(kv_mask)

        attn_output, _ = self.cross_attention(
            q, kv, kv, key_padding_mask=kv_key_pad_mask
        )
        out = self.norm1(q + self.dropout(attn_output))
        out = self.norm2(out + self.ffn(out))

        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1)

        return out


class CrossAttentionFusionWithConstraint(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim,
        hyperparams, R, num_heads=4, pool_method="attn"
    ):
        super().__init__()
        self.pool_method = pool_method
        self.fusion = CrossAttentionFusion(input_dim, num_heads)

        if pool_method == "attn":
            self.pool_query = nn.Parameter(torch.randn(1, 1, input_dim))
            self.pool_attn = nn.MultiheadAttention(
                input_dim, num_heads, batch_first=True
            )
            self.pool_norm = nn.LayerNorm(input_dim)

        self.constrained_model = ConstrainedFFNNModel(
            input_dim, hidden_dim, output_dim, hyperparams, R
        )

        self._pooling_frozen = False

    def freeze_pooling(self):
        self._pooling_frozen = True
        self.fusion.eval()
        for p in self.fusion.parameters():
            p.requires_grad = False
        if self.pool_method == "attn":
            self.pool_query.requires_grad = False
            self.pool_attn.eval()
            for p in self.pool_attn.parameters():
                p.requires_grad = False
            self.pool_norm.eval()
            for p in self.pool_norm.parameters():
                p.requires_grad = False

    def reinit_classifier(self):
        for layer in self.constrained_model.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def _reinit_module(self, module):
        if isinstance(module, nn.MultiheadAttention):
            module.out_proj.reset_parameters()
            module._reset_parameters()
        elif hasattr(module, '_reset_parameters'):
            module._reset_parameters()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            for child in module.children():
                self._reinit_module(child)

    def reinit_all(self):
        self._reinit_module(self.fusion.cross_attention)
        self._reinit_module(self.fusion.norm1)
        self._reinit_module(self.fusion.norm2)
        self._reinit_module(self.fusion.ffn)
        if self.pool_method == "attn":
            nn.init.normal_(self.pool_query)
            self._reinit_module(self.pool_attn)
            self._reinit_module(self.pool_norm)
        self.reinit_classifier()

    def train(self, mode=True):
        super().train(mode)
        if self._pooling_frozen:
            self.fusion.eval()
            if self.pool_method == "attn":
                self.pool_attn.eval()
                self.pool_norm.eval()
        return self

    def forward(self, q, kv, q_mask=None, kv_mask=None):
        fused = self.fusion(q, kv, q_mask=q_mask, kv_mask=kv_mask)

        if self.pool_method == "cls":
            pooled = fused[:, 0, :]
        elif self.pool_method == "mean":
            pooled = _masked_mean(fused, q_mask)
        elif self.pool_method == "attn":
            query = self.pool_query.expand(fused.size(0), -1, -1)
            q_key_pad_mask = _to_key_padding_mask(q_mask)
            attn_out, _ = self.pool_attn(
                query, fused, fused, key_padding_mask=q_key_pad_mask
            )
            pooled = self.pool_norm(attn_out).squeeze(1)
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")

        return self.constrained_model(pooled)


class SelfAttentionPooling(nn.Module):
    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

        self.pool_query = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pool_attn = nn.MultiheadAttention(
            emb_dim, num_heads, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        key_pad_mask = _to_key_padding_mask(mask)
        attn_output, _ = self.self_attention(
            x, x, x, key_padding_mask=key_pad_mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.ffn(x))

        query = self.pool_query.expand(x.size(0), -1, -1)
        attn_out, _ = self.pool_attn(
            query, x, x, key_padding_mask=key_pad_mask
        )
        pooled = self.pool_norm(attn_out).squeeze(1)
        return pooled

class SelfAttentionPoolingSinglePath(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R,
                 leaf_index, num_heads=4):
        super().__init__()
        self.pooling = SelfAttentionPooling(input_dim, num_heads)
        self.constrained_model = SinglePathFFNNModel(
            input_dim, hidden_dim, output_dim, hyperparams, R, leaf_index
        )
        self._pooling_frozen = False

    def freeze_pooling(self):
        self._pooling_frozen = True
        self.pooling.eval()
        for p in self.pooling.parameters():
            p.requires_grad = False

    def reinit_classifier(self):
        for layer in self.constrained_model.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def _reinit_module(self, module):
        if isinstance(module, nn.MultiheadAttention):
            module.out_proj.reset_parameters()
            module._reset_parameters()
        elif hasattr(module, '_reset_parameters'):
            module._reset_parameters()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            for child in module.children():
                self._reinit_module(child)

    def reinit_all(self):
        self._reinit_module(self.pooling.self_attention)
        self._reinit_module(self.pooling.norm1)
        self._reinit_module(self.pooling.norm2)
        self._reinit_module(self.pooling.ffn)
        nn.init.normal_(self.pooling.pool_query)
        self._reinit_module(self.pooling.pool_attn)
        self._reinit_module(self.pooling.pool_norm)
        self.reinit_classifier()

    def train(self, mode=True):
        super().train(mode)
        if self._pooling_frozen:
            self.pooling.eval()
        return self

    def forward(self, x, mask=None):
        pooled = self.pooling(x, mask=mask)
        return self.constrained_model(pooled)

class SelfAttentionPoolingWithConstraint(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim,
        hyperparams, R, num_heads=4
    ):
        super().__init__()
        self.pooling = SelfAttentionPooling(input_dim, num_heads)
        self.constrained_model = ConstrainedFFNNModel(
            input_dim, hidden_dim, output_dim, hyperparams, R
        )

        self._pooling_frozen = False

    def freeze_pooling(self):
        self._pooling_frozen = True
        self.pooling.eval()
        for p in self.pooling.parameters():
            p.requires_grad = False

    def reinit_classifier(self):
        for layer in self.constrained_model.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def _reinit_module(self, module):
        if isinstance(module, nn.MultiheadAttention):
            module.out_proj.reset_parameters()
            module._reset_parameters()
        elif hasattr(module, '_reset_parameters'):
            module._reset_parameters()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            for child in module.children():
                self._reinit_module(child)

    def reinit_all(self):
        self._reinit_module(self.pooling.self_attention)
        self._reinit_module(self.pooling.norm1)
        self._reinit_module(self.pooling.norm2)
        self._reinit_module(self.pooling.ffn)
        nn.init.normal_(self.pooling.pool_query)
        self._reinit_module(self.pooling.pool_attn)
        self._reinit_module(self.pooling.pool_norm)
        self.reinit_classifier()

    def train(self, mode=True):
        super().train(mode)
        if self._pooling_frozen:
            self.pooling.eval()
        return self

    def forward(self, x, mask=None):
        pooled = self.pooling(x, mask=mask)
        return self.constrained_model(pooled)

class ExpertHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams):
        super().__init__()
        self.nb_layers = hyperparams['num_layers']

        fc = []
        for i in range(self.nb_layers):
            if self.nb_layers == 1:
                fc.append(nn.Linear(input_dim, output_dim))
            elif i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(hyperparams['dropout'])

        if hyperparams['activation'] == 'tanh':
            self.f = nn.Tanh()
        elif hyperparams['activation'] == 'relu':
            self.f = nn.ReLU()
        else:
            self.f = nn.GELU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.fc[i](x)
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        return x


class FeatureRouter(nn.Module):
    def __init__(self, feature_dim, num_classes, R,
                 branch_hyperparams, branches, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.R = R

        self.expert_heads = nn.ModuleList([
            ExpertHead(feature_dim, feature_dim, num_classes, hp)
            for hp in branch_hyperparams
        ])

        for head, branch in zip(self.expert_heads, branches):
            head.fc.load_state_dict(branch.constrained_model.fc.state_dict())
        self._expert_heads_frozen = True
        self._freeze_expert_heads()

        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 3 * num_classes),
        )

        self.sigmoid = nn.Sigmoid()
        self._last_gate_weights = None

    def _freeze_expert_heads(self):
        self._expert_heads_frozen = True
        for head in self.expert_heads:
            head.eval()
            for p in head.parameters():
                p.requires_grad = False

    def _unfreeze_expert_heads(self):
        self._expert_heads_frozen = False
        for head in self.expert_heads:
            head.train()
            for p in head.parameters():
                p.requires_grad = True

    def reinit_expert_heads(self):
        for head in self.expert_heads:
            for layer in head.fc:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self._unfreeze_expert_heads()

    def train(self, mode=True):
        super().train(mode)
        if self._expert_heads_frozen:
            for head in self.expert_heads:
                head.eval()
        return self

    def forward(self, f_img, f_txt, f_cross):
        B = f_img.size(0)
        features = [f_img, f_txt, f_cross]

        expert_logits = torch.stack(
            [head(f) for head, f in zip(self.expert_heads, features)],
            dim=1,
        )

        concat = torch.cat(features, dim=-1)
        weights = self.gate(concat)
        weights = weights.view(B, 3, self.num_classes)
        weights = F.softmax(weights, dim=1)
        self._last_gate_weights = weights.detach()

        fused_logits = (expert_logits * weights).sum(dim=1)
        x = self.sigmoid(fused_logits)

        if self.training:
            return x
        else:
            constrained_out = get_constr_out(x, self.R)
            return x, constrained_out

class FeatureRouterSinglePath(nn.Module):
    def __init__(self, feature_dim, num_classes, R,
                 branch_hyperparams, branches, leaf_index, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.R = R
        self.register_buffer(
            "leaf_index", torch.as_tensor(leaf_index, dtype=torch.long)
        )

        self.expert_heads = nn.ModuleList([
            ExpertHead(feature_dim, feature_dim, num_classes, hp)
            for hp in branch_hyperparams
        ])

        for head, branch in zip(self.expert_heads, branches):
            head.fc.load_state_dict(branch.constrained_model.fc.state_dict())
        self._expert_heads_frozen = True
        self._freeze_expert_heads()

        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 3 * num_classes),
        )

        self._last_gate_weights = None

    def _freeze_expert_heads(self):
        self._expert_heads_frozen = True
        for head in self.expert_heads:
            head.eval()
            for p in head.parameters():
                p.requires_grad = False

    def _unfreeze_expert_heads(self):
        self._expert_heads_frozen = False
        for head in self.expert_heads:
            head.train()
            for p in head.parameters():
                p.requires_grad = True

    def reinit_expert_heads(self):
        for head in self.expert_heads:
            for layer in head.fc:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self._unfreeze_expert_heads()

    def train(self, mode=True):
        super().train(mode)
        if self._expert_heads_frozen:
            for head in self.expert_heads:
                head.eval()
        return self

    def forward(self, f_img, f_txt, f_cross):
        B = f_img.size(0)
        features = [f_img, f_txt, f_cross]

        expert_logits = torch.stack(
            [head(f) for head, f in zip(self.expert_heads, features)],
            dim=1,
        )

        concat = torch.cat(features, dim=-1)
        weights = self.gate(concat)
        weights = weights.view(B, 3, self.num_classes)
        weights = F.softmax(weights, dim=1)
        self._last_gate_weights = weights.detach()

        fused_logits = (expert_logits * weights).sum(dim=1)

        if self.training:
            return fused_logits
        else:
            coherent = softmax_constr_out(fused_logits, self.R, self.leaf_index)
            return fused_logits, coherent

class AdaptiveFusionWithConstraint(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_dim,
        img_branch,
        txt_branch,
        imgxtxt_branch,
        R,
        branch_hyperparams,
        dropout=0.1,
    ):
        super().__init__()

        self.img_branch = img_branch
        self.txt_branch = txt_branch
        self.imgxtxt_branch = imgxtxt_branch
        self._freeze_branches()

        self.router = FeatureRouter(
            feature_dim=feature_dim,
            num_classes=num_classes,
            R=R,
            branch_hyperparams=branch_hyperparams,
            branches=[img_branch, txt_branch, imgxtxt_branch],
            dropout=dropout,
        )

    def _freeze_branches(self):
        self._branches_frozen = True
        for branch in [self.img_branch, self.txt_branch, self.imgxtxt_branch]:
            branch.eval()
            for p in branch.parameters():
                p.requires_grad = False

    def _unfreeze_branches(self):
        self._branches_frozen = False
        for branch in [self.img_branch, self.txt_branch, self.imgxtxt_branch]:
            branch.train()
            for p in branch.parameters():
                p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self._branches_frozen:
            self.img_branch.eval()
            self.txt_branch.eval()
            self.imgxtxt_branch.eval()
        return self

    def _extract_unimodal_features(self, branch, x, mask):
        return branch.pooling(x, mask=mask)

    def _extract_cross_features(self, branch, q, kv, q_mask, kv_mask):
        fused = branch.fusion(q, kv, q_mask=q_mask, kv_mask=kv_mask)

        if branch.pool_method == "cls":
            pooled = fused[:, 0, :]
        elif branch.pool_method == "mean":
            pooled = _masked_mean(fused, q_mask)
        elif branch.pool_method == "attn":
            query = branch.pool_query.expand(fused.size(0), -1, -1)
            q_key_pad_mask = _to_key_padding_mask(q_mask)
            attn_out, _ = branch.pool_attn(
                query, fused, fused, key_padding_mask=q_key_pad_mask
            )
            pooled = branch.pool_norm(attn_out).squeeze(1)
        else:
            raise ValueError(f"Unknown pool method: {branch.pool_method}")

        return pooled

    def reinit_expert_heads(self):
        self.router.reinit_expert_heads()

    def reinit_router(self):
        self.reinit_expert_heads()
        for module in self.router.gate:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def reinit_all(self):
        self.img_branch.reinit_all()
        self.txt_branch.reinit_all()
        self.imgxtxt_branch.reinit_all()
        self._unfreeze_branches()
        self.reinit_router()

    def trainable_parameters(self):
        excluded = set()
        for branch in [self.img_branch, self.txt_branch, self.imgxtxt_branch]:
            excluded.update(id(p) for p in branch.constrained_model.parameters())
        return [p for p in self.parameters() if p.requires_grad and id(p) not in excluded]

    def forward(self, img, txt, img_mask=None, txt_mask=None):
        if self._branches_frozen:
            with torch.no_grad():
                f_img = self._extract_unimodal_features(
                    self.img_branch, img, img_mask
                )
                f_txt = self._extract_unimodal_features(
                    self.txt_branch, txt, txt_mask
                )
                f_cross = self._extract_cross_features(
                    self.imgxtxt_branch, img, txt, img_mask, txt_mask
                )
        else:
            f_img = self._extract_unimodal_features(
                self.img_branch, img, img_mask
            )
            f_txt = self._extract_unimodal_features(
                self.txt_branch, txt, txt_mask
            )
            f_cross = self._extract_cross_features(
                self.imgxtxt_branch, img, txt, img_mask, txt_mask
            )

        return self.router(f_img, f_txt, f_cross)

class AdaptiveFusionSinglePath(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_dim,
        img_branch,
        txt_branch,
        imgxtxt_branch,
        R,
        branch_hyperparams,
        leaf_index,
        dropout=0.1,
    ):
        super().__init__()

        self.img_branch = img_branch
        self.txt_branch = txt_branch
        self.imgxtxt_branch = imgxtxt_branch
        self._freeze_branches()

        self.router = FeatureRouterSinglePath(
            feature_dim=feature_dim,
            num_classes=num_classes,
            R=R,
            branch_hyperparams=branch_hyperparams,
            branches=[img_branch, txt_branch, imgxtxt_branch],
            leaf_index=leaf_index,
            dropout=dropout,
        )

    def _freeze_branches(self):
        self._branches_frozen = True
        for branch in [self.img_branch, self.txt_branch, self.imgxtxt_branch]:
            branch.eval()
            for p in branch.parameters():
                p.requires_grad = False

    def _unfreeze_branches(self):
        self._branches_frozen = False
        for branch in [self.img_branch, self.txt_branch, self.imgxtxt_branch]:
            branch.train()
            for p in branch.parameters():
                p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self._branches_frozen:
            self.img_branch.eval()
            self.txt_branch.eval()
            self.imgxtxt_branch.eval()
        return self

    def _extract_unimodal_features(self, branch, x, mask):
        return branch.pooling(x, mask=mask)

    def _extract_cross_features(self, branch, q, kv, q_mask, kv_mask):
        fused = branch.fusion(q, kv, q_mask=q_mask, kv_mask=kv_mask)
        if branch.pool_method == "cls":
            pooled = fused[:, 0, :]
        elif branch.pool_method == "mean":
            pooled = _masked_mean(fused, q_mask)
        elif branch.pool_method == "attn":
            query = branch.pool_query.expand(fused.size(0), -1, -1)
            q_key_pad_mask = _to_key_padding_mask(q_mask)
            attn_out, _ = branch.pool_attn(
                query, fused, fused, key_padding_mask=q_key_pad_mask
            )
            pooled = branch.pool_norm(attn_out).squeeze(1)
        else:
            raise ValueError(f"Unknown pool method: {branch.pool_method}")
        return pooled

    def reinit_expert_heads(self):
        self.router.reinit_expert_heads()

    def reinit_router(self):
        self.reinit_expert_heads()
        for module in self.router.gate:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def reinit_all(self):
        self.img_branch.reinit_all()
        self.txt_branch.reinit_all()
        self.imgxtxt_branch.reinit_all()
        self._unfreeze_branches()
        self.reinit_router()

    def trainable_parameters(self):
        excluded = set()
        for branch in [self.img_branch, self.txt_branch, self.imgxtxt_branch]:
            excluded.update(id(p) for p in branch.constrained_model.parameters())
        return [p for p in self.parameters() if p.requires_grad and id(p) not in excluded]

    def forward(self, img, txt, img_mask=None, txt_mask=None):
        if self._branches_frozen:
            with torch.no_grad():
                f_img = self._extract_unimodal_features(self.img_branch, img, img_mask)
                f_txt = self._extract_unimodal_features(self.txt_branch, txt, txt_mask)
                f_cross = self._extract_cross_features(
                    self.imgxtxt_branch, img, txt, img_mask, txt_mask
                )
        else:
            f_img = self._extract_unimodal_features(self.img_branch, img, img_mask)
            f_txt = self._extract_unimodal_features(self.txt_branch, txt, txt_mask)
            f_cross = self._extract_cross_features(
                self.imgxtxt_branch, img, txt, img_mask, txt_mask
            )
        return self.router(f_img, f_txt, f_cross)

class MLPClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams):
        super(MLPClassifier, self).__init__()

        self.nb_layers = hyperparams['num_layers']

        fc = []
        for i in range(self.nb_layers):
            if self.nb_layers == 1:
                fc.append(nn.Linear(input_dim, output_dim))
            elif i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(hyperparams['dropout'])

        if hyperparams['activation'] == 'tanh':
            self.f = nn.Tanh()
        elif hyperparams['activation'] == 'relu':
            self.f = nn.ReLU()
        else:
            self.f = nn.GELU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.fc[i](x)
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        return x


class SelfAttentionPoolingWithMLP(nn.Module):

    def __init__(
        self, input_dim, hidden_dim, output_dim,
        hyperparams, num_heads=4
    ):
        super().__init__()
        self.pooling = SelfAttentionPooling(input_dim, num_heads)
        self.classifier = MLPClassifier(
            input_dim, hidden_dim, output_dim, hyperparams
        )

        self._pooling_frozen = False

    def freeze_pooling(self):
        self._pooling_frozen = True
        self.pooling.eval()
        for p in self.pooling.parameters():
            p.requires_grad = False

    def reinit_classifier(self):
        for layer in self.classifier.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def _reinit_module(self, module):
        if isinstance(module, nn.MultiheadAttention):
            module.out_proj.reset_parameters()
            module._reset_parameters()
        elif hasattr(module, '_reset_parameters'):
            module._reset_parameters()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            for child in module.children():
                self._reinit_module(child)

    def reinit_all(self):
        self._reinit_module(self.pooling.self_attention)
        self._reinit_module(self.pooling.norm1)
        self._reinit_module(self.pooling.norm2)
        self._reinit_module(self.pooling.ffn)
        nn.init.normal_(self.pooling.pool_query)
        self._reinit_module(self.pooling.pool_attn)
        self._reinit_module(self.pooling.pool_norm)
        self.reinit_classifier()

    def train(self, mode=True):
        super().train(mode)
        if self._pooling_frozen:
            self.pooling.eval()
        return self

    def forward(self, x, mask=None):
        pooled = self.pooling(x, mask=mask)
        return self.classifier(pooled)


class CrossAttentionFusionWithMLP(nn.Module):

    def __init__(
        self, input_dim, hidden_dim, output_dim,
        hyperparams, num_heads=4, pool_method="attn"
    ):
        super().__init__()
        self.pool_method = pool_method
        self.fusion = CrossAttentionFusion(input_dim, num_heads)

        if pool_method == "attn":
            self.pool_query = nn.Parameter(torch.randn(1, 1, input_dim))
            self.pool_attn = nn.MultiheadAttention(
                input_dim, num_heads, batch_first=True
            )
            self.pool_norm = nn.LayerNorm(input_dim)

        self.classifier = MLPClassifier(
            input_dim, hidden_dim, output_dim, hyperparams
        )

        self._pooling_frozen = False

    def freeze_pooling(self):
        self._pooling_frozen = True
        self.fusion.eval()
        for p in self.fusion.parameters():
            p.requires_grad = False
        if self.pool_method == "attn":
            self.pool_query.requires_grad = False
            self.pool_attn.eval()
            for p in self.pool_attn.parameters():
                p.requires_grad = False
            self.pool_norm.eval()
            for p in self.pool_norm.parameters():
                p.requires_grad = False

    def reinit_classifier(self):
        for layer in self.classifier.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def _reinit_module(self, module):
        if isinstance(module, nn.MultiheadAttention):
            module.out_proj.reset_parameters()
            module._reset_parameters()
        elif hasattr(module, '_reset_parameters'):
            module._reset_parameters()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            for child in module.children():
                self._reinit_module(child)

    def reinit_all(self):
        self._reinit_module(self.fusion.cross_attention)
        self._reinit_module(self.fusion.norm1)
        self._reinit_module(self.fusion.norm2)
        self._reinit_module(self.fusion.ffn)
        if self.pool_method == "attn":
            nn.init.normal_(self.pool_query)
            self._reinit_module(self.pool_attn)
            self._reinit_module(self.pool_norm)
        self.reinit_classifier()

    def train(self, mode=True):
        super().train(mode)
        if self._pooling_frozen:
            self.fusion.eval()
            if self.pool_method == "attn":
                self.pool_attn.eval()
                self.pool_norm.eval()
        return self

    def forward(self, q, kv, q_mask=None, kv_mask=None):
        fused = self.fusion(q, kv, q_mask=q_mask, kv_mask=kv_mask)

        if self.pool_method == "cls":
            pooled = fused[:, 0, :]
        elif self.pool_method == "mean":
            pooled = _masked_mean(fused, q_mask)
        elif self.pool_method == "attn":
            query = self.pool_query.expand(fused.size(0), -1, -1)
            q_key_pad_mask = _to_key_padding_mask(q_mask)
            attn_out, _ = self.pool_attn(
                query, fused, fused, key_padding_mask=q_key_pad_mask
            )
            pooled = self.pool_norm(attn_out).squeeze(1)
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")

        return self.classifier(pooled)

class CrossAttentionFusionSinglePath(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R,
                 leaf_index, num_heads=4, pool_method="attn"):
        super().__init__()
        self.pool_method = pool_method
        self.fusion = CrossAttentionFusion(input_dim, num_heads)
        if pool_method == "attn":
            self.pool_query = nn.Parameter(torch.randn(1, 1, input_dim))
            self.pool_attn = nn.MultiheadAttention(
                input_dim, num_heads, batch_first=True
            )
            self.pool_norm = nn.LayerNorm(input_dim)
        self.constrained_model = SinglePathFFNNModel(
            input_dim, hidden_dim, output_dim, hyperparams, R, leaf_index
        )
        self._pooling_frozen = False

    def freeze_pooling(self):
        self._pooling_frozen = True
        self.fusion.eval()
        for p in self.fusion.parameters():
            p.requires_grad = False
        if self.pool_method == "attn":
            self.pool_query.requires_grad = False
            self.pool_attn.eval()
            for p in self.pool_attn.parameters():
                p.requires_grad = False
            self.pool_norm.eval()
            for p in self.pool_norm.parameters():
                p.requires_grad = False

    def reinit_classifier(self):
        for layer in self.constrained_model.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def _reinit_module(self, module):
        if isinstance(module, nn.MultiheadAttention):
            module.out_proj.reset_parameters()
            module._reset_parameters()
        elif hasattr(module, '_reset_parameters'):
            module._reset_parameters()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            for child in module.children():
                self._reinit_module(child)

    def reinit_all(self):
        self._reinit_module(self.fusion.cross_attention)
        self._reinit_module(self.fusion.norm1)
        self._reinit_module(self.fusion.norm2)
        self._reinit_module(self.fusion.ffn)
        if self.pool_method == "attn":
            nn.init.normal_(self.pool_query)
            self._reinit_module(self.pool_attn)
            self._reinit_module(self.pool_norm)
        self.reinit_classifier()

    def train(self, mode=True):
        super().train(mode)
        if self._pooling_frozen:
            self.fusion.eval()
            if self.pool_method == "attn":
                self.pool_attn.eval()
                self.pool_norm.eval()
        return self

    def forward(self, q, kv, q_mask=None, kv_mask=None):
        fused = self.fusion(q, kv, q_mask=q_mask, kv_mask=kv_mask)
        if self.pool_method == "cls":
            pooled = fused[:, 0, :]
        elif self.pool_method == "mean":
            pooled = _masked_mean(fused, q_mask)
        elif self.pool_method == "attn":
            query = self.pool_query.expand(fused.size(0), -1, -1)
            q_key_pad_mask = _to_key_padding_mask(q_mask)
            attn_out, _ = self.pool_attn(
                query, fused, fused, key_padding_mask=q_key_pad_mask
            )
            pooled = self.pool_norm(attn_out).squeeze(1)
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
        return self.constrained_model(pooled)


class MLPExpertHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams):
        super().__init__()
        self.nb_layers = hyperparams['num_layers']

        fc = []
        for i in range(self.nb_layers):
            if self.nb_layers == 1:
                fc.append(nn.Linear(input_dim, output_dim))
            elif i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(hyperparams['dropout'])

        if hyperparams['activation'] == 'tanh':
            self.f = nn.Tanh()
        elif hyperparams['activation'] == 'relu':
            self.f = nn.ReLU()
        else:
            self.f = nn.GELU()

    def forward(self, x):
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                x = self.fc[i](x)
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)
        return x


class FeatureRouterMLP(nn.Module):
    def __init__(self, feature_dim, num_classes,
                 branch_hyperparams, branches, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes

        self.expert_heads = nn.ModuleList([
            MLPExpertHead(feature_dim, feature_dim, num_classes, hp)
            for hp in branch_hyperparams
        ])

        for head, branch in zip(self.expert_heads, branches):
            head.fc.load_state_dict(branch.classifier.fc.state_dict())
        self._expert_heads_frozen = True
        self._freeze_expert_heads()

        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 3 * num_classes),
        )

        self._last_gate_weights = None

    def _freeze_expert_heads(self):
        self._expert_heads_frozen = True
        for head in self.expert_heads:
            head.eval()
            for p in head.parameters():
                p.requires_grad = False

    def _unfreeze_expert_heads(self):
        self._expert_heads_frozen = False
        for head in self.expert_heads:
            head.train()
            for p in head.parameters():
                p.requires_grad = True

    def reinit_expert_heads(self):
        for head in self.expert_heads:
            for layer in head.fc:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self._unfreeze_expert_heads()

    def train(self, mode=True):
        super().train(mode)
        if self._expert_heads_frozen:
            for head in self.expert_heads:
                head.eval()
        return self

    def forward(self, f_img, f_txt, f_cross):
        B = f_img.size(0)
        features = [f_img, f_txt, f_cross]

        expert_logits = torch.stack(
            [head(f) for head, f in zip(self.expert_heads, features)],
            dim=1,
        )

        concat = torch.cat(features, dim=-1)
        weights = self.gate(concat)
        weights = weights.view(B, 3, self.num_classes)
        weights = F.softmax(weights, dim=1)
        self._last_gate_weights = weights.detach()

        fused_logits = (expert_logits * weights).sum(dim=1)

        return fused_logits


class AdaptiveFusionWithMLP(nn.Module):
    def __init__(
        self,
        num_classes,
        feature_dim,
        img_branch,
        txt_branch,
        imgxtxt_branch,
        branch_hyperparams,
        dropout=0.1,
    ):
        super().__init__()

        self.img_branch = img_branch
        self.txt_branch = txt_branch
        self.imgxtxt_branch = imgxtxt_branch
        self._freeze_branches()

        self.router = FeatureRouterMLP(
            feature_dim=feature_dim,
            num_classes=num_classes,
            branch_hyperparams=branch_hyperparams,
            branches=[img_branch, txt_branch, imgxtxt_branch],
            dropout=dropout,
        )

    def _freeze_branches(self):
        self._branches_frozen = True
        for branch in [self.img_branch, self.txt_branch, self.imgxtxt_branch]:
            branch.eval()
            for p in branch.parameters():
                p.requires_grad = False

    def _unfreeze_branches(self):
        self._branches_frozen = False
        for branch in [self.img_branch, self.txt_branch, self.imgxtxt_branch]:
            branch.train()
            for p in branch.parameters():
                p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self._branches_frozen:
            self.img_branch.eval()
            self.txt_branch.eval()
            self.imgxtxt_branch.eval()
        return self

    def _extract_unimodal_features(self, branch, x, mask):
        return branch.pooling(x, mask=mask)

    def _extract_cross_features(self, branch, q, kv, q_mask, kv_mask):
        fused = branch.fusion(q, kv, q_mask=q_mask, kv_mask=kv_mask)

        if branch.pool_method == "cls":
            pooled = fused[:, 0, :]
        elif branch.pool_method == "mean":
            pooled = _masked_mean(fused, q_mask)
        elif branch.pool_method == "attn":
            query = branch.pool_query.expand(fused.size(0), -1, -1)
            q_key_pad_mask = _to_key_padding_mask(q_mask)
            attn_out, _ = branch.pool_attn(
                query, fused, fused, key_padding_mask=q_key_pad_mask
            )
            pooled = branch.pool_norm(attn_out).squeeze(1)
        else:
            raise ValueError(f"Unknown pool method: {branch.pool_method}")

        return pooled

    def reinit_expert_heads(self):
        self.router.reinit_expert_heads()

    def reinit_router(self):
        self.reinit_expert_heads()
        for module in self.router.gate:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def reinit_all(self):
        self.img_branch.reinit_all()
        self.txt_branch.reinit_all()
        self.imgxtxt_branch.reinit_all()
        self._unfreeze_branches()
        self.reinit_router()

    def trainable_parameters(self):
        excluded = set()
        for branch in [self.img_branch, self.txt_branch, self.imgxtxt_branch]:
            excluded.update(id(p) for p in branch.classifier.parameters())
        return [p for p in self.parameters() if p.requires_grad and id(p) not in excluded]

    def forward(self, img, txt, img_mask=None, txt_mask=None):
        if self._branches_frozen:
            with torch.no_grad():
                f_img = self._extract_unimodal_features(
                    self.img_branch, img, img_mask
                )
                f_txt = self._extract_unimodal_features(
                    self.txt_branch, txt, txt_mask
                )
                f_cross = self._extract_cross_features(
                    self.imgxtxt_branch, img, txt, img_mask, txt_mask
                )
        else:
            f_img = self._extract_unimodal_features(
                self.img_branch, img, img_mask
            )
            f_txt = self._extract_unimodal_features(
                self.txt_branch, txt, txt_mask
            )
            f_cross = self._extract_cross_features(
                self.imgxtxt_branch, img, txt, img_mask, txt_mask
            )

        return self.router(f_img, f_txt, f_cross)


def load_branch_mlp(checkpoint_path, branch_type, input_dim, num_classes, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt["hyperparams"]
    saved_args = ckpt.get("args", {})

    if branch_type == "unimodal":
        branch = SelfAttentionPoolingWithMLP(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=num_classes,
            hyperparams=hp,
            num_heads=saved_args.get("num_heads", 4),
        )
    elif branch_type == "cross":
        branch = CrossAttentionFusionWithMLP(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=num_classes,
            hyperparams=hp,
            num_heads=saved_args.get("num_heads", 4),
            pool_method=saved_args.get("pool_method", "attn"),
        )
    else:
        raise ValueError(f"Unknown branch_type: {branch_type}")

    branch.load_state_dict(ckpt["model_state_dict"])
    branch.to(device)
    return branch, hp
