import torch
import torchvision.models as models
from vision_transformer_pytorch import VisionTransformer  # Ensure this is your custom implementation
from utils import checkpoint  # Assuming this is a utility for saving models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Instantiate the Official and Custom Models
official_vit = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT).to(device)
official_vit.eval()  # Set to evaluation mode

my_vit = VisionTransformer(
    image_h=224,
    image_w=224,
    image_c=3,
    patch_d=32,
    dropout=0.2,
    n_encoders=12,
    n_heads=12,
    embedding_dim=768,
    ff_multiplier=4,
    n_classes=2,
    device=device
).to(device)

# 2. Extract State Dictionaries
official_sd = official_vit.state_dict()
my_sd = my_vit.state_dict()

# 3. Initialize a New State Dictionary for the Custom Model
new_sd = {}

for k, v in official_sd.items():
    if k.startswith('encoder.layers'):
        # Example key: 'encoder.layers.encoder_layer_0.ln_1.weight'
        parts = k.split('.')
        layer_num = int(parts[2].split('_')[-1])  # Extract layer index
        layer_component = parts[3]  # e.g., 'ln_1', 'self_attention', 'mlp'

        if layer_component.startswith('ln'):
            # Mapping LayerNorm layers
            norm_num = layer_component.split('_')[1]  # '1' or '2'
            param = parts[4]  # 'weight' or 'bias'
            new_key = f'encoder_stack.layer_list.{layer_num}.norm{norm_num}.{param}'
            new_sd[new_key] = v

        elif layer_component == 'self_attention':
            # Mapping Multi-Head Attention layers
            proj_type = parts[4]  # e.g., 'in_proj_weight', 'out_proj.weight'

            if proj_type == 'in_proj_weight':
                # Official MHA consolidates Q, K, V; split them
                q_weight, k_weight, v_weight = v.chunk(3, dim=0)
                new_sd[f'encoder_stack.layer_list.{layer_num}.mhsa.q_proj.weight'] = q_weight
                new_sd[f'encoder_stack.layer_list.{layer_num}.mhsa.k_proj.weight'] = k_weight
                new_sd[f'encoder_stack.layer_list.{layer_num}.mhsa.v_proj.weight'] = v_weight

            elif proj_type == 'in_proj_bias':
                q_bias, k_bias, v_bias = v.chunk(3, dim=0)
                new_sd[f'encoder_stack.layer_list.{layer_num}.mhsa.q_proj.bias'] = q_bias
                new_sd[f'encoder_stack.layer_list.{layer_num}.mhsa.k_proj.bias'] = k_bias
                new_sd[f'encoder_stack.layer_list.{layer_num}.mhsa.v_proj.bias'] = v_bias

            elif proj_type == 'out_proj.weight':
                new_key = f'encoder_stack.layer_list.{layer_num}.mhsa.linear_out.weight'
                new_sd[new_key] = v

            elif proj_type == 'out_proj.bias':
                new_key = f'encoder_stack.layer_list.{layer_num}.mhsa.linear_out.bias'
                new_sd[new_key] = v

        elif layer_component == 'mlp':
            # Mapping Feed-Forward Networks
            mlp_layer = parts[4]  # '0' or '3'
            param = parts[5]  # 'weight' or 'bias'

            if mlp_layer == '0':
                new_key = f'encoder_stack.layer_list.{layer_num}.feed_forward.0.{param}'
            elif mlp_layer == '3':
                new_key = f'encoder_stack.layer_list.{layer_num}.feed_forward.2.{param}'

            new_sd[new_key] = v

    # Mapping Patch Embedding (Conv2d) and Position Embedding
    elif k == 'conv_proj.weight':
        # Official ViT's Conv2d weight shape: (embed_dim, in_channels, patch_size, patch_size)
        # Custom ViT's Linear weight shape: (embed_dim, in_channels * patch_size * patch_size)
        # Reshape Conv2d weights to match Linear layer
        embed_dim = v.shape[0]
        in_channels = v.shape[1]
        patch_size = v.shape[2]
        # Flatten the Conv2d weights
        linear_weight = v.view(embed_dim, in_channels * patch_size * patch_size)
        new_sd['encoder_stack.patch_embedding.patch_embedding_layer.weight'] = linear_weight
    elif k == 'conv_proj.bias':
        new_sd['encoder_stack.patch_embedding.patch_embedding_layer.bias'] = v
    elif k == 'encoder.pos_embedding':
        # Official ViT's pos_embedding shape: (1, num_patches + 1, embed_dim)
        # Custom ViT's position_embedding shape: (1, num_patches + 1, embed_dim) including class token
        pos_embed = v  # Shape: (1, N+1, D)
        # Assign entire pos_embedding to custom model's position_embedding
        # Assuming your custom model has a class token integrated within position_embedding
        new_sd['encoder_stack.patch_embedding.position_embedding.weight'] = pos_embed.squeeze(0)
    elif k == 'class_token':
        # Mapping the class token
        new_sd['encoder_stack.patch_embedding.class_token.weight'] = v
    elif k == 'encoder.ln.weight':
        new_sd['encoder_stack.norm.weight'] = v
    elif k == 'encoder.ln.bias':
        new_sd['encoder_stack.norm.bias'] = v
    # Optionally, map other components if necessary

# 4. Update the Custom Model's State Dictionary
# This ensures that only matching keys are updated, and others remain unchanged
my_sd.update(new_sd)

# 5. Load the Updated State Dictionary into the Custom Model
missing_keys, unexpected_keys = my_vit.load_state_dict(my_sd, strict=False)

if missing_keys:
    print("Missing keys in the custom model after loading:", missing_keys)
if unexpected_keys:
    print("Unexpected keys in the custom model after loading:", unexpected_keys)

# 6. Verify the Loading (Optional)
# You can perform a forward pass with dummy data to ensure the model works
dummy_input = torch.randn(1, 3, 224, 224).to(device)
output = my_vit(dummy_input)
print("Output shape:", output.shape)  # Should be [1, 2] based on n_classes=2

# 7. Save the Weights in New Format
checkpoint(model=my_vit, filename=f"models/cat_dog/cat_dog_image_net.pth")
