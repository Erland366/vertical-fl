"""
CLIPModel(
  (text_model): CLIPTextTransformer(
    (embeddings): CLIPTextEmbeddings(
      (token_embedding): Embedding(49408, 512)
      (position_embedding): Embedding(77, 512)
    )
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-11): 12 x CLIPEncoderLayer(
          (self_attn): CLIPSdpaAttention(
            (k_proj): Linear(in_features=512, out_features=512, bias=True)
            (v_proj): Linear(in_features=512, out_features=512, bias=True)
            (q_proj): Linear(in_features=512, out_features=512, bias=True)
            (out_proj): Linear(in_features=512, out_features=512, bias=True)
          )
          (layer_norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): QuickGELUActivation()
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
          )
          (layer_norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
  (vision_model): CLIPVisionTransformer(
    (embeddings): CLIPVisionEmbeddings(
      (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)
      (position_embedding): Embedding(197, 768)
    )
    (pre_layrnorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-11): 12 x CLIPEncoderLayer(
          (self_attn): CLIPSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): QuickGELUActivation()
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
          )
          (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (visual_projection): Linear(in_features=768, out_features=512, bias=False)
  (text_projection): Linear(in_features=512, out_features=512, bias=False)
)
"""
from transformers import CLIPModel

from torch import nn

class CLIPTextClient(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.text_model = clip_model.text_model

    def forward(self, input_ids, attention_mask):
        text_outputs = self.text_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return text_outputs.last_hidden_state[:, 0, :] # CLIP uses the first token as the CLS token

class CLIPImageClient(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.vision_model = clip_model.vision_model

    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values, output_hidden_states=True)
        return vision_outputs.pooler_output

class CLIPServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.visual_projection = clip_model.visual_projection
        self.text_projection = clip_model.text_projection

        self.logit_scale = clip_model.logit_scale
            
    def forward(self, image_embeddings, text_embeddings):
        image_embeddings = self.visual_projection(image_embeddings) # [B, 512]
        text_embeddings = self.text_projection(text_embeddings) # [B, 512]

        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text