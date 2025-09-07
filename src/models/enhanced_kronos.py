import torch
import torch.nn as nn
import torch.nn.functional as F
from models.kronos import Kronos, KronosTokenizer
from models.module import TransformerBlock, RMSNorm


class EnhancedKronosTokenizer(nn.Module):
    """
    增强版的KronosTokenizer，添加了Dropout和BatchNorm正则化层
    """
    def __init__(self, base_tokenizer, dropout_rate=0.3, use_batch_norm=True):
        super().__init__()
        self.base_tokenizer = base_tokenizer
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 在关键位置添加Dropout层
        self.input_dropout = nn.Dropout(dropout_rate)
        self.encoder_dropout = nn.Dropout(dropout_rate * 0.8)  # 编码器使用稍低的dropout
        self.decoder_dropout = nn.Dropout(dropout_rate * 0.8)  # 解码器使用稍低的dropout
        self.quant_dropout = nn.Dropout(dropout_rate * 0.5)   # 量化层使用更低的dropout
        self.output_dropout = nn.Dropout(dropout_rate * 0.3)   # 输出层使用最低的dropout
        
        # 添加BatchNorm层
        if use_batch_norm:
            self.input_batch_norm = nn.BatchNorm1d(base_tokenizer.d_model)
            self.encoder_batch_norm = nn.ModuleList([
                nn.BatchNorm1d(base_tokenizer.d_model) for _ in range(len(base_tokenizer.encoder))
            ])
            self.decoder_batch_norm = nn.ModuleList([
                nn.BatchNorm1d(base_tokenizer.d_model) for _ in range(len(base_tokenizer.decoder))
            ])
            self.quant_batch_norm = nn.BatchNorm1d(base_tokenizer.codebook_dim)
            self.post_quant_batch_norm = nn.BatchNorm1d(base_tokenizer.d_model)
        
    def forward(self, x):
        """
        增强版的前向传播，添加了Dropout和BatchNorm
        """
        # 输入处理
        z = self.base_tokenizer.embed(x)
        z = self.input_dropout(z)
        
        # 添加BatchNorm（需要调整维度）
        if self.use_batch_norm:
            batch_size, seq_len, d_model = z.shape
            z = z.view(-1, d_model)  # (batch_size * seq_len, d_model)
            z = self.input_batch_norm(z)
            z = z.view(batch_size, seq_len, d_model)
        
        # 编码器层
        for i, layer in enumerate(self.base_tokenizer.encoder):
            z = layer(z)
            z = self.encoder_dropout(z)
            if self.use_batch_norm:
                batch_size, seq_len, d_model = z.shape
                z = z.view(-1, d_model)
                z = self.encoder_batch_norm[i](z)
                z = z.view(batch_size, seq_len, d_model)
        
        # 量化前处理
        z = self.base_tokenizer.quant_embed(z)
        z = self.quant_dropout(z)
        
        if self.use_batch_norm:
            batch_size, seq_len, codebook_dim = z.shape
            z = z.view(-1, codebook_dim)
            z = self.quant_batch_norm(z)
            z = z.view(batch_size, seq_len, codebook_dim)
        
        # 量化过程
        bsq_loss, quantized, z_indices = self.base_tokenizer.tokenizer(z)
        
        # 量化后处理
        quantized_pre = quantized[:, :, :self.base_tokenizer.s1_bits]
        z_pre = self.base_tokenizer.post_quant_embed_pre(quantized_pre)
        z = self.base_tokenizer.post_quant_embed(quantized)
        
        if self.use_batch_norm:
            batch_size, seq_len, d_model = z.shape
            z = z.view(-1, d_model)
            z = self.post_quant_batch_norm(z)
            z = z.view(batch_size, seq_len, d_model)
            
            z_pre = z_pre.view(-1, d_model)
            z_pre = self.post_quant_batch_norm(z_pre)
            z_pre = z_pre.view(batch_size, seq_len, d_model)
        
        # 解码器层
        for i, layer in enumerate(self.base_tokenizer.decoder):
            z_pre = layer(z_pre)
            z_pre = self.decoder_dropout(z_pre)
            if self.use_batch_norm:
                batch_size, seq_len, d_model = z_pre.shape
                z_pre = z_pre.view(-1, d_model)
                z_pre = self.decoder_batch_norm[i](z_pre)
                z_pre = z_pre.view(batch_size, seq_len, d_model)
        
        z_pre = self.base_tokenizer.head(z_pre)
        z_pre = self.output_dropout(z_pre)
        
        # Apply head to z as well to ensure proper dimension projection
        z = self.base_tokenizer.head(z)
        z = self.output_dropout(z)
        
        return (z_pre, z), bsq_loss, quantized, z_indices
    
    def save_pretrained(self, save_directory):
        """保存增强模型"""
        self.base_tokenizer.save_pretrained(save_directory)
        # 保存增强配置
        import json
        config = {
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        }
        with open(f"{save_directory}/enhanced_config.json", 'w') as f:
            json.dump(config, f)


class EnhancedKronos(nn.Module):
    """
    增强版的Kronos模型，添加了Dropout和BatchNorm正则化层
    """
    def __init__(self, base_model, dropout_rate=0.3, use_batch_norm=True):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 在关键位置添加Dropout层 - 优化dropout率分配
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.6)  # 降低嵌入层dropout
        self.transformer_dropout = nn.Dropout(dropout_rate * 0.8)  # 提高transformer层dropout
        self.dependency_dropout = nn.Dropout(dropout_rate * 0.4)  # 依赖层dropout
        self.head_dropout = nn.Dropout(dropout_rate * 0.2)  # 降低head层dropout
        
        # 添加LayerNorm作为BatchNorm的替代或补充
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(base_model.d_model) for _ in range(len(base_model.transformer))
        ])
        
        # 添加BatchNorm层 - 优化BatchNorm配置
        if use_batch_norm:
            self.embedding_batch_norm = nn.BatchNorm1d(base_model.d_model, momentum=0.1)
            self.transformer_batch_norm = nn.ModuleList([
                nn.BatchNorm1d(base_model.d_model, momentum=0.1) for _ in range(len(base_model.transformer))
            ])
            self.dependency_batch_norm = nn.BatchNorm1d(base_model.d_model, momentum=0.1)
            
        # 添加残差连接的投影层
        self.residual_projections = nn.ModuleList([
            nn.Linear(base_model.d_model, base_model.d_model) 
            for _ in range(len(base_model.transformer))
        ])
        
        # 添加注意力增强机制
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.gating_network = nn.Sequential(
            nn.Linear(base_model.d_model, base_model.d_model // 4),
            nn.ReLU(),
            nn.Linear(base_model.d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        """
        增强版的前向传播，添加了Dropout、BatchNorm、残差连接和注意力增强
        """
        # 嵌入层
        x = self.base_model.embedding([s1_ids, s2_ids])
        
        # 时间嵌入
        if stamp is not None:
            time_embedding = self.base_model.time_emb(stamp)
            x = x + time_embedding
        
        # 应用Dropout和BatchNorm
        x = self.embedding_dropout(x)
        if self.use_batch_norm:
            batch_size, seq_len, d_model = x.shape
            x = x.view(-1, d_model)
            x = self.embedding_batch_norm(x)
            x = x.view(batch_size, seq_len, d_model)
        
        # Transformer层 - 增强版本
        for i, layer in enumerate(self.base_model.transformer):
            # 保存残差连接
            residual = x
            
            # 应用transformer层
            x = layer(x, key_padding_mask=padding_mask)
            
            # 应用注意力增强
            attention_gate = self.gating_network(x.mean(dim=1, keepdim=True))
            x = x * self.attention_scale * attention_gate + x * (1 - attention_gate)
            
            # 应用残差连接
            x = x + self.residual_projections[i](residual)
            
            # 应用Dropout
            x = self.transformer_dropout(x)
            
            # 应用LayerNorm（总是使用）
            x = self.layer_norms[i](x)
            
            # 应用BatchNorm（如果启用）
            if self.use_batch_norm:
                batch_size, seq_len, d_model = x.shape
                x = x.view(-1, d_model)
                x = self.transformer_batch_norm[i](x)
                x = x.view(batch_size, seq_len, d_model)
        
        # 最终归一化
        x = self.base_model.norm(x)
        
        # S1预测 - 增强版本
        s1_logits = self.base_model.head(x)
        s1_logits = self.head_dropout(s1_logits)
        
        # S2预测 - 增强版本
        if use_teacher_forcing:
            sibling_embed = self.base_model.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            # 使用top-k采样提高多样性
            top_k = min(50, self.base_model.s1_vocab_size)
            s1_probs_topk, topk_indices = torch.topk(s1_probs, top_k, dim=-1)
            s1_probs_topk = s1_probs_topk / s1_probs_topk.sum(dim=-1, keepdim=True)
            sample_indices = torch.multinomial(s1_probs_topk.view(-1, top_k), 1).view(s1_ids.shape)
            sample_s1_ids = topk_indices.gather(-1, sample_indices.unsqueeze(-1)).squeeze(-1)
            sibling_embed = self.base_model.embedding.emb_s1(sample_s1_ids)
        
        # 依赖层处理
        x2 = self.base_model.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        x2 = self.dependency_dropout(x2)
        
        # 应用BatchNorm
        if self.use_batch_norm:
            batch_size, seq_len, d_model = x2.shape
            x2 = x2.view(-1, d_model)
            x2 = self.dependency_batch_norm(x2)
            x2 = x2.view(batch_size, seq_len, d_model)
        
        s2_logits = self.base_model.head.cond_forward(x2)
        s2_logits = self.head_dropout(s2_logits)
        
        return s1_logits, s2_logits
    
    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """增强版的S1解码"""
        x = self.base_model.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.base_model.time_emb(stamp)
            x = x + time_embedding
        
        x = self.embedding_dropout(x)
        if self.use_batch_norm:
            batch_size, seq_len, d_model = x.shape
            x = x.view(-1, d_model)
            x = self.embedding_batch_norm(x)
            x = x.view(batch_size, seq_len, d_model)
        
        for i, layer in enumerate(self.base_model.transformer):
            x = layer(x, key_padding_mask=padding_mask)
            x = self.transformer_dropout(x)
            if self.use_batch_norm:
                batch_size, seq_len, d_model = x.shape
                x = x.view(-1, d_model)
                x = self.transformer_batch_norm[i](x)
                x = x.view(batch_size, seq_len, d_model)
        
        x = self.base_model.norm(x)
        s1_logits = self.base_model.head(x)
        s1_logits = self.head_dropout(s1_logits)
        
        return s1_logits, x
    
    def decode_s2(self, context, s1_ids, padding_mask=None):
        """增强版的S2解码"""
        sibling_embed = self.base_model.embedding.emb_s1(s1_ids)
        x2 = self.base_model.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        x2 = self.dependency_dropout(x2)
        
        if self.use_batch_norm:
            batch_size, seq_len, d_model = x2.shape
            x2 = x2.view(-1, d_model)
            x2 = self.dependency_batch_norm(x2)
            x2 = x2.view(batch_size, seq_len, d_model)
        
        s2_logits = self.base_model.head.cond_forward(x2)
        s2_logits = self.head_dropout(s2_logits)
        
        return s2_logits
    
    def save_pretrained(self, save_directory):
        """保存增强模型"""
        self.base_model.save_pretrained(save_directory)
        # 保存增强配置
        import json
        config = {
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        }
        with open(f"{save_directory}/enhanced_config.json", 'w') as f:
            json.dump(config, f)


def create_enhanced_tokenizer(base_tokenizer_path, dropout_rate=0.3, use_batch_norm=True):
    """
    创建增强版的tokenizer
    """
    # 检查输入是否为路径字符串或模型对象
    if isinstance(base_tokenizer_path, str):
        base_tokenizer = KronosTokenizer.from_pretrained(base_tokenizer_path)
    else:
        # 如果传入的是模型对象，直接使用
        base_tokenizer = base_tokenizer_path
    
    enhanced_tokenizer = EnhancedKronosTokenizer(base_tokenizer, dropout_rate, use_batch_norm)
    return enhanced_tokenizer


def create_enhanced_model(base_model_path, dropout_rate=0.3, use_batch_norm=True):
    """
    创建增强版的Kronos模型
    """
    # 检查输入是否为路径字符串或模型对象
    if isinstance(base_model_path, str):
        base_model = Kronos.from_pretrained(base_model_path)
    else:
        # 如果传入的是模型对象，直接使用
        base_model = base_model_path
    
    enhanced_model = EnhancedKronos(base_model, dropout_rate, use_batch_norm)
    return enhanced_model