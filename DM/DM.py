import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


warnings.filterwarnings('once',
                        message="Attention mask is not 2D, this is not intended to happen, please check your model.")


class TokenRouter(nn.Module):
    def __init__(self, embed_dim,max_tokens):
        super().__init__()
        self.weight_predictor = nn.Linear(embed_dim, 1)


        self.k_predictor = nn.Sequential(
                        nn.Linear(max_tokens * embed_dim, 512),  
                        nn.LeakyReLU(),                
                        nn.Linear(512, 1)          
                            )

        self.max_tokens = max_tokens
        self.aux_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 2)  
        )
        self.aux_optimizer = torch.optim.Adam(self.aux_predictor.parameters(), lr=1e-4)

    def forward(self, x):
        original_type = x.dtype
        self.weight_predictor.to(torch.float32)
        self.k_predictor.to(torch.float32)

    
        weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype)).squeeze(-1)  
 
        x_flattened = x.view(x.size(0), -1)  
        k_logits = self.k_predictor(x_flattened.to(self.k_predictor[0].weight.dtype)) 
        k = torch.sigmoid(k_logits).squeeze(-1) * self.max_tokens  
        k = torch.clamp(k, 1, self.max_tokens).long()  

        return weights.to(original_type), k

    def aux_prediction(self, x):
        logits = self.aux_predictor(x)
        return logits
      


class DM(nn.Module):
    def __init__(self, block, aux_loss_on=True,max_tokens=360):
        super().__init__()
        self.router = TokenRouter(256,max_tokens=max_tokens)
        self.block = block
        self.training_step = 0
        self.aux_loss_on = aux_loss_on

    def forward(self,
                x,
                attention_mask,
                past_key_values,
                output_attentions,
                use_cache,
                head_mask,
                cache_position=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                **kwargs):
        b, s, d = x.shape
            
        if self.training:
            weights ,k_values = self.router(x)
            selected_mask = []

            for i in range(b):
                k = k_values[i].item()
                top_k_values, _ = torch.topk(weights[i], k, sorted=True)
                threshold = top_k_values[-1]
                current_selected_mask = weights[i] > threshold
                selected_mask.append(current_selected_mask)


            selected_mask = torch.stack(selected_mask)  

            logits = self.router.aux_prediction(x.detach()).view(-1, 2)
            selected_mask_labels = selected_mask.long().view(-1)   
            loss_aux = F.cross_entropy(logits, selected_mask_labels)  
            

            processed_tokens = torch.zeros_like(x)

      
            for i in range(b):

                current_selected_mask = selected_mask[i]
                selected_tokens = x[i][current_selected_mask]

                if attention_mask is not None:
                    current_causal_mask = attention_mask[i, 0, 0]
                    current_causal_mask = current_causal_mask[current_selected_mask].unsqueeze(0).unsqueeze(0)

                if selected_tokens.size(0) > 0:
                    processed_tokens[i][current_selected_mask] = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=current_causal_mask,#1,1,59
                        layer_past=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        head_mask=head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        **kwargs)[0].squeeze(0) * weights[i][selected_mask[i]].unsqueeze(-1)
            


            output = processed_tokens + (x * (~selected_mask).unsqueeze(-1).to(x.dtype))
      

            self.router.aux_optimizer.zero_grad()
            loss_aux.backward()
            self.router.aux_optimizer.step()#

        else: 
            weights,k_values =self.router(x)
   
            logits = self.router.aux_prediction(x)
            selected_mask = torch.argmax(logits, dim=-1).bool()


            processed_tokens = torch.zeros_like(x)

            for i in range(b):
                current_selected_mask = selected_mask[i]
                selected_tokens = x[i][current_selected_mask]

                if attention_mask is not None:
                    current_causal_mask = attention_mask[i, 0, 0]
                    current_causal_mask = current_causal_mask[current_selected_mask].unsqueeze(0).unsqueeze(0)

                if selected_tokens.size(0) > 0:
                    processed_tokens[i][current_selected_mask] = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=current_causal_mask,
                        layer_past=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        head_mask=head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        **kwargs)[0].squeeze(0) * weights[i][selected_mask[i]].unsqueeze(-1)

            output = processed_tokens + (x * (~selected_mask).unsqueeze(-1).to(x.dtype))
      

        return output, sum(selected_mask.float().sum(dim=1)) / b











class DM_no_aux(nn.Module):
    def __init__(self, block, aux_loss_on=True,max_tokens=60):
        super().__init__()
        self.router = TokenRouter(256,max_tokens=max_tokens)
        self.block = block
        self.training_step = 0

    def forward(self,
                x,
                attention_mask,
                past_key_values,
                output_attentions,
                use_cache,
                head_mask,
                cache_position=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                **kwargs):
        b, s, d = x.shape
            
        if self.training:
            weights,threshold = self.router(x)
            selected_mask = []

            selected_mask = (weights > threshold.unsqueeze(-1)) 
            

            processed_tokens = torch.zeros_like(x)

            for i in range(b):

                current_selected_mask = selected_mask[i]
                selected_tokens = x[i][current_selected_mask]

                if attention_mask is not None:
                    current_causal_mask = attention_mask[i, 0, 0]
                    current_causal_mask = current_causal_mask[current_selected_mask].unsqueeze(0).unsqueeze(0)

                if selected_tokens.size(0) > 0:
                    processed_tokens[i][current_selected_mask] = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=current_causal_mask,#1,1,59
                        layer_past=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        head_mask=head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        **kwargs)[0].squeeze(0) * weights[i][selected_mask[i]].unsqueeze(-1)
            

            output=processed_tokens+x


        else: 
            weights, threshold = self.router(x)
            selected_mask = []
            selected_mask = (weights > threshold.unsqueeze(-1)) 
            processed_tokens = torch.zeros_like(x)

            for i in range(b):
                current_selected_mask = selected_mask[i]
                selected_tokens = x[i][current_selected_mask]  

                if attention_mask is not None:
                    current_causal_mask = attention_mask[i, 0, 0]
                    current_causal_mask = current_causal_mask[current_selected_mask].unsqueeze(0).unsqueeze(0)

                if selected_tokens.size(0) > 0:
                    processed_tokens[i][current_selected_mask] = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=current_causal_mask,
                        layer_past=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        head_mask=head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        **kwargs)[0].squeeze(0) * weights[i][selected_mask[i]].unsqueeze(-1)


            output=processed_tokens+x

    
        return output, None




















class Router(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight_predictor = nn.Linear(embed_dim, 1)

        self.aux_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 2)  
        )
        self.aux_optimizer = torch.optim.Adam(self.aux_predictor.parameters(), lr=1e-4)

    def forward(self, x):
        original_type = x.dtype
        self.weight_predictor.to(torch.float32)
        weights = self.weight_predictor(x.to(self.weight_predictor.weight.dtype)).squeeze(-1)  # [batch_size, seq_len] 
        return weights.to(original_type)

    def aux_prediction(self, x):
        logits = self.aux_predictor(x)
        return logits





class DM_fixed_k(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.router = Router(256)
        self.block = block
        self.capacity = 0.5

    def forward(self,
                x,
                attention_mask,
                past_key_values,
                output_attentions,
                use_cache,
                head_mask,
                cache_position=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                **kwargs):
        b, s, d = x.shape#128，60，768
            
        if self.training:
            weights = self.router(x)#64,60



            k = max(1, int(self.capacity * s))
            top_k_values, _ = torch.topk(weights, k, dim=1, sorted=True)
            threshold = top_k_values[:, -1]
            selected_mask = weights > threshold.unsqueeze(-1) if k > 1 else weights >= threshold.unsqueeze(-1)

            cache = None

            processed_tokens = torch.zeros_like(x)

       
            for i in range(b):

                current_selected_mask = selected_mask[i]
                selected_tokens = x[i][current_selected_mask]

                if attention_mask is not None:
                    current_causal_mask = attention_mask[i, 0, 0]
                    current_causal_mask = current_causal_mask[current_selected_mask].unsqueeze(0).unsqueeze(0)

                if selected_tokens.size(0) > 0:
                    processed_tokens[i][current_selected_mask] = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=current_causal_mask,#1,1,59
                        layer_past=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        head_mask=head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        **kwargs)[0].squeeze(0) * weights[i][selected_mask[i]].unsqueeze(-1)
            
            output=processed_tokens+x


        else: 
            weights= self.router(x)

            logits = self.router.aux_prediction(x)
            selected_mask = torch.argmax(logits, dim=-1).bool()

            processed_tokens = torch.zeros_like(x)

            for i in range(b):
                current_selected_mask = selected_mask[i]
                selected_tokens = x[i][current_selected_mask]  

                if attention_mask is not None:
                    current_causal_mask = attention_mask[i, 0, 0]
                    current_causal_mask = current_causal_mask[current_selected_mask].unsqueeze(0).unsqueeze(0)

                if selected_tokens.size(0) > 0:
                    processed_tokens[i][current_selected_mask] = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=current_causal_mask,#1,1,59
                        layer_past=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        head_mask=head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        **kwargs)[0].squeeze(0) * weights[i][selected_mask[i]].unsqueeze(-1)

            output=processed_tokens+x

        return output, None
