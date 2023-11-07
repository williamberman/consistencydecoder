import torch
from diffusers import StableDiffusionPipeline
from consistencydecoder import ConsistencyDecoder, save_image, load_image
from diffusers.models.unet_2d_blocks import ResnetDownsampleBlock2D, UNetMidBlock2D, ResnetUpsampleBlock2D

from conv_unet_vae import ConvUNetVAE, rename_state_dict
from safetensors.torch import load_file as stl
from torch import nn
import ipdb

# encode with stable diffusion vae
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe.vae.cuda()

# construct original decoder with jitted model
decoder_consistency = ConsistencyDecoder(device="cuda:0")

# construct UNet code, overwrite the decoder with conv_unet_vae
model = ConvUNetVAE()
model.load_state_dict(
    rename_state_dict(
        stl("consistency_decoder.safetensors"),
        stl("embedding.safetensors"),
    )
)
model = model.cuda()

decoder_consistency.ckpt = model

# image = load_image("test_dog_image.jpg", size=(256, 256), center_crop=True)
image = load_image("assets/gt1.png", size=(256, 256), center_crop=True)
latent = pipe.vae.encode(image.half().cuda()).latent_dist.sample()

# decode with gan
sample_gan = pipe.vae.decode(latent).sample.detach()
save_image(sample_gan, "gan.png")

# decode with conv_unet_vae
sample_consistency_orig = decoder_consistency(latent)
save_image(sample_consistency_orig, "con_orig.png")


########### conversion

print('CONVERSION')

print('DOWN BLOCK ONE')

block_one_sd_orig = model.down[0].state_dict()
block_one_sd_new = {}

for i in range(3):
    block_one_sd_new[f"resnets.{i}.norm1.weight"]         = block_one_sd_orig.pop(f"{i}.gn_1.weight")
    block_one_sd_new[f"resnets.{i}.norm1.bias"]           = block_one_sd_orig.pop(f"{i}.gn_1.bias")
    block_one_sd_new[f"resnets.{i}.conv1.weight"]         = block_one_sd_orig.pop(f"{i}.f_1.weight")
    block_one_sd_new[f"resnets.{i}.conv1.bias"]           = block_one_sd_orig.pop(f"{i}.f_1.bias")
    block_one_sd_new[f"resnets.{i}.time_emb_proj.weight"] = block_one_sd_orig.pop(f"{i}.f_t.weight")
    block_one_sd_new[f"resnets.{i}.time_emb_proj.bias"]   = block_one_sd_orig.pop(f"{i}.f_t.bias")
    block_one_sd_new[f"resnets.{i}.norm2.weight"]         = block_one_sd_orig.pop(f"{i}.gn_2.weight")
    block_one_sd_new[f"resnets.{i}.norm2.bias"]           = block_one_sd_orig.pop(f"{i}.gn_2.bias")
    block_one_sd_new[f"resnets.{i}.conv2.weight"]         = block_one_sd_orig.pop(f"{i}.f_2.weight")
    block_one_sd_new[f"resnets.{i}.conv2.bias"]           = block_one_sd_orig.pop(f"{i}.f_2.bias")

block_one_sd_new[f"downsamplers.0.norm1.weight"]         = block_one_sd_orig.pop(f"3.gn_1.weight")
block_one_sd_new[f"downsamplers.0.norm1.bias"]           = block_one_sd_orig.pop(f"3.gn_1.bias")
block_one_sd_new[f"downsamplers.0.conv1.weight"]         = block_one_sd_orig.pop(f"3.f_1.weight")
block_one_sd_new[f"downsamplers.0.conv1.bias"]           = block_one_sd_orig.pop(f"3.f_1.bias")
block_one_sd_new[f"downsamplers.0.time_emb_proj.weight"] = block_one_sd_orig.pop(f"3.f_t.weight")
block_one_sd_new[f"downsamplers.0.time_emb_proj.bias"]   = block_one_sd_orig.pop(f"3.f_t.bias")
block_one_sd_new[f"downsamplers.0.norm2.weight"]         = block_one_sd_orig.pop(f"3.gn_2.weight")
block_one_sd_new[f"downsamplers.0.norm2.bias"]           = block_one_sd_orig.pop(f"3.gn_2.bias")
block_one_sd_new[f"downsamplers.0.conv2.weight"]         = block_one_sd_orig.pop(f"3.f_2.weight")
block_one_sd_new[f"downsamplers.0.conv2.bias"]           = block_one_sd_orig.pop(f"3.f_2.bias")

assert len(block_one_sd_orig) == 0

block_one = ResnetDownsampleBlock2D(
    in_channels=320, 
    out_channels=320, 
    temb_channels=1280, 
    num_layers=3, 
    add_downsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5
)

block_one.load_state_dict(block_one_sd_new)

print('DOWN BLOCK TWO')

block_two_sd_orig = model.down[1].state_dict()
block_two_sd_new = {}

for i in range(3):
    block_two_sd_new[f"resnets.{i}.norm1.weight"]         = block_two_sd_orig.pop(f"{i}.gn_1.weight")
    block_two_sd_new[f"resnets.{i}.norm1.bias"]           = block_two_sd_orig.pop(f"{i}.gn_1.bias")
    block_two_sd_new[f"resnets.{i}.conv1.weight"]         = block_two_sd_orig.pop(f"{i}.f_1.weight")
    block_two_sd_new[f"resnets.{i}.conv1.bias"]           = block_two_sd_orig.pop(f"{i}.f_1.bias")
    block_two_sd_new[f"resnets.{i}.time_emb_proj.weight"] = block_two_sd_orig.pop(f"{i}.f_t.weight")
    block_two_sd_new[f"resnets.{i}.time_emb_proj.bias"]   = block_two_sd_orig.pop(f"{i}.f_t.bias")
    block_two_sd_new[f"resnets.{i}.norm2.weight"]         = block_two_sd_orig.pop(f"{i}.gn_2.weight")
    block_two_sd_new[f"resnets.{i}.norm2.bias"]           = block_two_sd_orig.pop(f"{i}.gn_2.bias")
    block_two_sd_new[f"resnets.{i}.conv2.weight"]         = block_two_sd_orig.pop(f"{i}.f_2.weight")
    block_two_sd_new[f"resnets.{i}.conv2.bias"]           = block_two_sd_orig.pop(f"{i}.f_2.bias")

    if i == 0:
        block_two_sd_new[f"resnets.{i}.conv_shortcut.weight"]           = block_two_sd_orig.pop(f"{i}.f_s.weight")
        block_two_sd_new[f"resnets.{i}.conv_shortcut.bias"]           = block_two_sd_orig.pop(f"{i}.f_s.bias")

block_two_sd_new[f"downsamplers.0.norm1.weight"]         = block_two_sd_orig.pop(f"3.gn_1.weight")
block_two_sd_new[f"downsamplers.0.norm1.bias"]           = block_two_sd_orig.pop(f"3.gn_1.bias")
block_two_sd_new[f"downsamplers.0.conv1.weight"]         = block_two_sd_orig.pop(f"3.f_1.weight")
block_two_sd_new[f"downsamplers.0.conv1.bias"]           = block_two_sd_orig.pop(f"3.f_1.bias")
block_two_sd_new[f"downsamplers.0.time_emb_proj.weight"] = block_two_sd_orig.pop(f"3.f_t.weight")
block_two_sd_new[f"downsamplers.0.time_emb_proj.bias"]   = block_two_sd_orig.pop(f"3.f_t.bias")
block_two_sd_new[f"downsamplers.0.norm2.weight"]         = block_two_sd_orig.pop(f"3.gn_2.weight")
block_two_sd_new[f"downsamplers.0.norm2.bias"]           = block_two_sd_orig.pop(f"3.gn_2.bias")
block_two_sd_new[f"downsamplers.0.conv2.weight"]         = block_two_sd_orig.pop(f"3.f_2.weight")
block_two_sd_new[f"downsamplers.0.conv2.bias"]           = block_two_sd_orig.pop(f"3.f_2.bias")

assert len(block_two_sd_orig) == 0

block_two = ResnetDownsampleBlock2D(
    in_channels=320, 
    out_channels=640, 
    temb_channels=1280, 
    num_layers=3, 
    add_downsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5
)

block_two.load_state_dict(block_two_sd_new)

print('DOWN BLOCK THREE')

block_three_sd_orig = model.down[2].state_dict()
block_three_sd_new = {}

for i in range(3):
    block_three_sd_new[f"resnets.{i}.norm1.weight"]         = block_three_sd_orig.pop(f"{i}.gn_1.weight")
    block_three_sd_new[f"resnets.{i}.norm1.bias"]           = block_three_sd_orig.pop(f"{i}.gn_1.bias")
    block_three_sd_new[f"resnets.{i}.conv1.weight"]         = block_three_sd_orig.pop(f"{i}.f_1.weight")
    block_three_sd_new[f"resnets.{i}.conv1.bias"]           = block_three_sd_orig.pop(f"{i}.f_1.bias")
    block_three_sd_new[f"resnets.{i}.time_emb_proj.weight"] = block_three_sd_orig.pop(f"{i}.f_t.weight")
    block_three_sd_new[f"resnets.{i}.time_emb_proj.bias"]   = block_three_sd_orig.pop(f"{i}.f_t.bias")
    block_three_sd_new[f"resnets.{i}.norm2.weight"]         = block_three_sd_orig.pop(f"{i}.gn_2.weight")
    block_three_sd_new[f"resnets.{i}.norm2.bias"]           = block_three_sd_orig.pop(f"{i}.gn_2.bias")
    block_three_sd_new[f"resnets.{i}.conv2.weight"]         = block_three_sd_orig.pop(f"{i}.f_2.weight")
    block_three_sd_new[f"resnets.{i}.conv2.bias"]           = block_three_sd_orig.pop(f"{i}.f_2.bias")

    if i == 0:
        block_three_sd_new[f"resnets.{i}.conv_shortcut.weight"]           = block_three_sd_orig.pop(f"{i}.f_s.weight")
        block_three_sd_new[f"resnets.{i}.conv_shortcut.bias"]           = block_three_sd_orig.pop(f"{i}.f_s.bias")

block_three_sd_new[f"downsamplers.0.norm1.weight"]         = block_three_sd_orig.pop(f"3.gn_1.weight")
block_three_sd_new[f"downsamplers.0.norm1.bias"]           = block_three_sd_orig.pop(f"3.gn_1.bias")
block_three_sd_new[f"downsamplers.0.conv1.weight"]         = block_three_sd_orig.pop(f"3.f_1.weight")
block_three_sd_new[f"downsamplers.0.conv1.bias"]           = block_three_sd_orig.pop(f"3.f_1.bias")
block_three_sd_new[f"downsamplers.0.time_emb_proj.weight"] = block_three_sd_orig.pop(f"3.f_t.weight")
block_three_sd_new[f"downsamplers.0.time_emb_proj.bias"]   = block_three_sd_orig.pop(f"3.f_t.bias")
block_three_sd_new[f"downsamplers.0.norm2.weight"]         = block_three_sd_orig.pop(f"3.gn_2.weight")
block_three_sd_new[f"downsamplers.0.norm2.bias"]           = block_three_sd_orig.pop(f"3.gn_2.bias")
block_three_sd_new[f"downsamplers.0.conv2.weight"]         = block_three_sd_orig.pop(f"3.f_2.weight")
block_three_sd_new[f"downsamplers.0.conv2.bias"]           = block_three_sd_orig.pop(f"3.f_2.bias")

assert len(block_three_sd_orig) == 0

block_three = ResnetDownsampleBlock2D(
    in_channels=640, 
    out_channels=1024, 
    temb_channels=1280, 
    num_layers=3, 
    add_downsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5
)

block_three.load_state_dict(block_three_sd_new)

print('DOWN BLOCK FOUR')

block_four_sd_orig = model.down[3].state_dict()
block_four_sd_new = {}

for i in range(3):
    block_four_sd_new[f"resnets.{i}.norm1.weight"]         = block_four_sd_orig.pop(f"{i}.gn_1.weight")
    block_four_sd_new[f"resnets.{i}.norm1.bias"]           = block_four_sd_orig.pop(f"{i}.gn_1.bias")
    block_four_sd_new[f"resnets.{i}.conv1.weight"]         = block_four_sd_orig.pop(f"{i}.f_1.weight")
    block_four_sd_new[f"resnets.{i}.conv1.bias"]           = block_four_sd_orig.pop(f"{i}.f_1.bias")
    block_four_sd_new[f"resnets.{i}.time_emb_proj.weight"] = block_four_sd_orig.pop(f"{i}.f_t.weight")
    block_four_sd_new[f"resnets.{i}.time_emb_proj.bias"]   = block_four_sd_orig.pop(f"{i}.f_t.bias")
    block_four_sd_new[f"resnets.{i}.norm2.weight"]         = block_four_sd_orig.pop(f"{i}.gn_2.weight")
    block_four_sd_new[f"resnets.{i}.norm2.bias"]           = block_four_sd_orig.pop(f"{i}.gn_2.bias")
    block_four_sd_new[f"resnets.{i}.conv2.weight"]         = block_four_sd_orig.pop(f"{i}.f_2.weight")
    block_four_sd_new[f"resnets.{i}.conv2.bias"]           = block_four_sd_orig.pop(f"{i}.f_2.bias")

assert len(block_four_sd_orig) == 0

block_four = ResnetDownsampleBlock2D(
    in_channels=1024, 
    out_channels=1024, 
    temb_channels=1280, 
    num_layers=3, 
    add_downsample=False,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5
)

block_four.load_state_dict(block_four_sd_new)


print('MID BLOCK 1')

mid_block_one_sd_orig = model.mid.state_dict()
mid_block_one_sd_new = {}

for i in range(2):
    mid_block_one_sd_new[f"resnets.{i}.norm1.weight"]         = mid_block_one_sd_orig.pop(f"{i}.gn_1.weight")
    mid_block_one_sd_new[f"resnets.{i}.norm1.bias"]           = mid_block_one_sd_orig.pop(f"{i}.gn_1.bias")
    mid_block_one_sd_new[f"resnets.{i}.conv1.weight"]         = mid_block_one_sd_orig.pop(f"{i}.f_1.weight")
    mid_block_one_sd_new[f"resnets.{i}.conv1.bias"]           = mid_block_one_sd_orig.pop(f"{i}.f_1.bias")
    mid_block_one_sd_new[f"resnets.{i}.time_emb_proj.weight"] = mid_block_one_sd_orig.pop(f"{i}.f_t.weight")
    mid_block_one_sd_new[f"resnets.{i}.time_emb_proj.bias"]   = mid_block_one_sd_orig.pop(f"{i}.f_t.bias")
    mid_block_one_sd_new[f"resnets.{i}.norm2.weight"]         = mid_block_one_sd_orig.pop(f"{i}.gn_2.weight")
    mid_block_one_sd_new[f"resnets.{i}.norm2.bias"]           = mid_block_one_sd_orig.pop(f"{i}.gn_2.bias")
    mid_block_one_sd_new[f"resnets.{i}.conv2.weight"]         = mid_block_one_sd_orig.pop(f"{i}.f_2.weight")
    mid_block_one_sd_new[f"resnets.{i}.conv2.bias"]           = mid_block_one_sd_orig.pop(f"{i}.f_2.bias")

assert len(mid_block_one_sd_orig) == 0

mid_block_one = UNetMidBlock2D(
    in_channels=1024, 
    temb_channels=1280, 
    num_layers=1, 
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5,
    add_attention=False,
)

mid_block_one.load_state_dict(mid_block_one_sd_new)

print('UP BLOCK ONE')

up_block_one_sd_orig = model.up[-1].state_dict()
up_block_one_sd_new = {}

for i in range(4):
    up_block_one_sd_new[f"resnets.{i}.norm1.weight"]         = up_block_one_sd_orig.pop(f"{i}.gn_1.weight")
    up_block_one_sd_new[f"resnets.{i}.norm1.bias"]           = up_block_one_sd_orig.pop(f"{i}.gn_1.bias")
    up_block_one_sd_new[f"resnets.{i}.conv1.weight"]         = up_block_one_sd_orig.pop(f"{i}.f_1.weight")
    up_block_one_sd_new[f"resnets.{i}.conv1.bias"]           = up_block_one_sd_orig.pop(f"{i}.f_1.bias")
    up_block_one_sd_new[f"resnets.{i}.time_emb_proj.weight"] = up_block_one_sd_orig.pop(f"{i}.f_t.weight")
    up_block_one_sd_new[f"resnets.{i}.time_emb_proj.bias"]   = up_block_one_sd_orig.pop(f"{i}.f_t.bias")
    up_block_one_sd_new[f"resnets.{i}.norm2.weight"]         = up_block_one_sd_orig.pop(f"{i}.gn_2.weight")
    up_block_one_sd_new[f"resnets.{i}.norm2.bias"]           = up_block_one_sd_orig.pop(f"{i}.gn_2.bias")
    up_block_one_sd_new[f"resnets.{i}.conv2.weight"]         = up_block_one_sd_orig.pop(f"{i}.f_2.weight")
    up_block_one_sd_new[f"resnets.{i}.conv2.bias"]           = up_block_one_sd_orig.pop(f"{i}.f_2.bias")
    up_block_one_sd_new[f"resnets.{i}.conv_shortcut.weight"] = up_block_one_sd_orig.pop(f"{i}.f_s.weight")
    up_block_one_sd_new[f"resnets.{i}.conv_shortcut.bias"]   = up_block_one_sd_orig.pop(f"{i}.f_s.bias")

up_block_one_sd_new[f"upsamplers.0.norm1.weight"]         = up_block_one_sd_orig.pop(f"4.gn_1.weight")
up_block_one_sd_new[f"upsamplers.0.norm1.bias"]           = up_block_one_sd_orig.pop(f"4.gn_1.bias")
up_block_one_sd_new[f"upsamplers.0.conv1.weight"]         = up_block_one_sd_orig.pop(f"4.f_1.weight")
up_block_one_sd_new[f"upsamplers.0.conv1.bias"]           = up_block_one_sd_orig.pop(f"4.f_1.bias")
up_block_one_sd_new[f"upsamplers.0.time_emb_proj.weight"] = up_block_one_sd_orig.pop(f"4.f_t.weight")
up_block_one_sd_new[f"upsamplers.0.time_emb_proj.bias"]   = up_block_one_sd_orig.pop(f"4.f_t.bias")
up_block_one_sd_new[f"upsamplers.0.norm2.weight"]         = up_block_one_sd_orig.pop(f"4.gn_2.weight")
up_block_one_sd_new[f"upsamplers.0.norm2.bias"]           = up_block_one_sd_orig.pop(f"4.gn_2.bias")
up_block_one_sd_new[f"upsamplers.0.conv2.weight"]         = up_block_one_sd_orig.pop(f"4.f_2.weight")
up_block_one_sd_new[f"upsamplers.0.conv2.bias"]           = up_block_one_sd_orig.pop(f"4.f_2.bias")

assert len(up_block_one_sd_orig) == 0

up_block_one = ResnetUpsampleBlock2D(
    in_channels=1024, 
    prev_output_channel=1024,
    out_channels=1024, 
    temb_channels=1280, 
    num_layers=4, 
    add_upsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5
)

up_block_one.load_state_dict(up_block_one_sd_new)

print('UP BLOCK TWO')

up_block_two_sd_orig = model.up[-2].state_dict()
up_block_two_sd_new = {}

for i in range(4):
    up_block_two_sd_new[f"resnets.{i}.norm1.weight"]         = up_block_two_sd_orig.pop(f"{i}.gn_1.weight")
    up_block_two_sd_new[f"resnets.{i}.norm1.bias"]           = up_block_two_sd_orig.pop(f"{i}.gn_1.bias")
    up_block_two_sd_new[f"resnets.{i}.conv1.weight"]         = up_block_two_sd_orig.pop(f"{i}.f_1.weight")
    up_block_two_sd_new[f"resnets.{i}.conv1.bias"]           = up_block_two_sd_orig.pop(f"{i}.f_1.bias")
    up_block_two_sd_new[f"resnets.{i}.time_emb_proj.weight"] = up_block_two_sd_orig.pop(f"{i}.f_t.weight")
    up_block_two_sd_new[f"resnets.{i}.time_emb_proj.bias"]   = up_block_two_sd_orig.pop(f"{i}.f_t.bias")
    up_block_two_sd_new[f"resnets.{i}.norm2.weight"]         = up_block_two_sd_orig.pop(f"{i}.gn_2.weight")
    up_block_two_sd_new[f"resnets.{i}.norm2.bias"]           = up_block_two_sd_orig.pop(f"{i}.gn_2.bias")
    up_block_two_sd_new[f"resnets.{i}.conv2.weight"]         = up_block_two_sd_orig.pop(f"{i}.f_2.weight")
    up_block_two_sd_new[f"resnets.{i}.conv2.bias"]           = up_block_two_sd_orig.pop(f"{i}.f_2.bias")
    up_block_two_sd_new[f"resnets.{i}.conv_shortcut.weight"] = up_block_two_sd_orig.pop(f"{i}.f_s.weight")
    up_block_two_sd_new[f"resnets.{i}.conv_shortcut.bias"]   = up_block_two_sd_orig.pop(f"{i}.f_s.bias")

up_block_two_sd_new[f"upsamplers.0.norm1.weight"]         = up_block_two_sd_orig.pop(f"4.gn_1.weight")
up_block_two_sd_new[f"upsamplers.0.norm1.bias"]           = up_block_two_sd_orig.pop(f"4.gn_1.bias")
up_block_two_sd_new[f"upsamplers.0.conv1.weight"]         = up_block_two_sd_orig.pop(f"4.f_1.weight")
up_block_two_sd_new[f"upsamplers.0.conv1.bias"]           = up_block_two_sd_orig.pop(f"4.f_1.bias")
up_block_two_sd_new[f"upsamplers.0.time_emb_proj.weight"] = up_block_two_sd_orig.pop(f"4.f_t.weight")
up_block_two_sd_new[f"upsamplers.0.time_emb_proj.bias"]   = up_block_two_sd_orig.pop(f"4.f_t.bias")
up_block_two_sd_new[f"upsamplers.0.norm2.weight"]         = up_block_two_sd_orig.pop(f"4.gn_2.weight")
up_block_two_sd_new[f"upsamplers.0.norm2.bias"]           = up_block_two_sd_orig.pop(f"4.gn_2.bias")
up_block_two_sd_new[f"upsamplers.0.conv2.weight"]         = up_block_two_sd_orig.pop(f"4.f_2.weight")
up_block_two_sd_new[f"upsamplers.0.conv2.bias"]           = up_block_two_sd_orig.pop(f"4.f_2.bias")

assert len(up_block_two_sd_orig) == 0

up_block_two = ResnetUpsampleBlock2D(
    in_channels=640, 
    prev_output_channel=1024,
    out_channels=1024, 
    temb_channels=1280, 
    num_layers=4, 
    add_upsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5
)

up_block_two.load_state_dict(up_block_two_sd_new)

print('UP BLOCK THREE')

up_block_three_sd_orig = model.up[-3].state_dict()
up_block_three_sd_new = {}

for i in range(4):
    up_block_three_sd_new[f"resnets.{i}.norm1.weight"]         = up_block_three_sd_orig.pop(f"{i}.gn_1.weight")
    up_block_three_sd_new[f"resnets.{i}.norm1.bias"]           = up_block_three_sd_orig.pop(f"{i}.gn_1.bias")
    up_block_three_sd_new[f"resnets.{i}.conv1.weight"]         = up_block_three_sd_orig.pop(f"{i}.f_1.weight")
    up_block_three_sd_new[f"resnets.{i}.conv1.bias"]           = up_block_three_sd_orig.pop(f"{i}.f_1.bias")
    up_block_three_sd_new[f"resnets.{i}.time_emb_proj.weight"] = up_block_three_sd_orig.pop(f"{i}.f_t.weight")
    up_block_three_sd_new[f"resnets.{i}.time_emb_proj.bias"]   = up_block_three_sd_orig.pop(f"{i}.f_t.bias")
    up_block_three_sd_new[f"resnets.{i}.norm2.weight"]         = up_block_three_sd_orig.pop(f"{i}.gn_2.weight")
    up_block_three_sd_new[f"resnets.{i}.norm2.bias"]           = up_block_three_sd_orig.pop(f"{i}.gn_2.bias")
    up_block_three_sd_new[f"resnets.{i}.conv2.weight"]         = up_block_three_sd_orig.pop(f"{i}.f_2.weight")
    up_block_three_sd_new[f"resnets.{i}.conv2.bias"]           = up_block_three_sd_orig.pop(f"{i}.f_2.bias")
    up_block_three_sd_new[f"resnets.{i}.conv_shortcut.weight"] = up_block_three_sd_orig.pop(f"{i}.f_s.weight")
    up_block_three_sd_new[f"resnets.{i}.conv_shortcut.bias"]   = up_block_three_sd_orig.pop(f"{i}.f_s.bias")

up_block_three_sd_new[f"upsamplers.0.norm1.weight"]         = up_block_three_sd_orig.pop(f"4.gn_1.weight")
up_block_three_sd_new[f"upsamplers.0.norm1.bias"]           = up_block_three_sd_orig.pop(f"4.gn_1.bias")
up_block_three_sd_new[f"upsamplers.0.conv1.weight"]         = up_block_three_sd_orig.pop(f"4.f_1.weight")
up_block_three_sd_new[f"upsamplers.0.conv1.bias"]           = up_block_three_sd_orig.pop(f"4.f_1.bias")
up_block_three_sd_new[f"upsamplers.0.time_emb_proj.weight"] = up_block_three_sd_orig.pop(f"4.f_t.weight")
up_block_three_sd_new[f"upsamplers.0.time_emb_proj.bias"]   = up_block_three_sd_orig.pop(f"4.f_t.bias")
up_block_three_sd_new[f"upsamplers.0.norm2.weight"]         = up_block_three_sd_orig.pop(f"4.gn_2.weight")
up_block_three_sd_new[f"upsamplers.0.norm2.bias"]           = up_block_three_sd_orig.pop(f"4.gn_2.bias")
up_block_three_sd_new[f"upsamplers.0.conv2.weight"]         = up_block_three_sd_orig.pop(f"4.f_2.weight")
up_block_three_sd_new[f"upsamplers.0.conv2.bias"]           = up_block_three_sd_orig.pop(f"4.f_2.bias")

assert len(up_block_three_sd_orig) == 0

up_block_three = ResnetUpsampleBlock2D(
    in_channels=320, 
    prev_output_channel=1024,
    out_channels=640, 
    temb_channels=1280, 
    num_layers=4, 
    add_upsample=True,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5
)

up_block_three.load_state_dict(up_block_three_sd_new)

print('UP BLOCK FOUR')

up_block_four_sd_orig = model.up[-4].state_dict()
up_block_four_sd_new = {}

for i in range(4):
    up_block_four_sd_new[f"resnets.{i}.norm1.weight"]         = up_block_four_sd_orig.pop(f"{i}.gn_1.weight")
    up_block_four_sd_new[f"resnets.{i}.norm1.bias"]           = up_block_four_sd_orig.pop(f"{i}.gn_1.bias")
    up_block_four_sd_new[f"resnets.{i}.conv1.weight"]         = up_block_four_sd_orig.pop(f"{i}.f_1.weight")
    up_block_four_sd_new[f"resnets.{i}.conv1.bias"]           = up_block_four_sd_orig.pop(f"{i}.f_1.bias")
    up_block_four_sd_new[f"resnets.{i}.time_emb_proj.weight"] = up_block_four_sd_orig.pop(f"{i}.f_t.weight")
    up_block_four_sd_new[f"resnets.{i}.time_emb_proj.bias"]   = up_block_four_sd_orig.pop(f"{i}.f_t.bias")
    up_block_four_sd_new[f"resnets.{i}.norm2.weight"]         = up_block_four_sd_orig.pop(f"{i}.gn_2.weight")
    up_block_four_sd_new[f"resnets.{i}.norm2.bias"]           = up_block_four_sd_orig.pop(f"{i}.gn_2.bias")
    up_block_four_sd_new[f"resnets.{i}.conv2.weight"]         = up_block_four_sd_orig.pop(f"{i}.f_2.weight")
    up_block_four_sd_new[f"resnets.{i}.conv2.bias"]           = up_block_four_sd_orig.pop(f"{i}.f_2.bias")
    up_block_four_sd_new[f"resnets.{i}.conv_shortcut.weight"] = up_block_four_sd_orig.pop(f"{i}.f_s.weight")
    up_block_four_sd_new[f"resnets.{i}.conv_shortcut.bias"]   = up_block_four_sd_orig.pop(f"{i}.f_s.bias")

assert len(up_block_four_sd_orig) == 0

up_block_four = ResnetUpsampleBlock2D(
    in_channels=320, 
    prev_output_channel=640,
    out_channels=320, 
    temb_channels=1280, 
    num_layers=4, 
    add_upsample=False,
    resnet_time_scale_shift="scale_shift",
    resnet_eps=1e-5
)

up_block_four.load_state_dict(up_block_four_sd_new)

print("initial projection (conv_in)")

conv_in_sd_orig = model.embed_image.state_dict()
conv_in_sd_new = {}

conv_in_sd_new['weight'] = conv_in_sd_orig.pop('f.weight')
conv_in_sd_new['bias'] = conv_in_sd_orig.pop('f.bias')

assert len(conv_in_sd_orig) == 0

in_channels = 7
block_out_channels = [320]
conv_in_kernel = 3
conv_in_padding = (conv_in_kernel - 1) // 2
conv_in = nn.Conv2d(
    in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
)

conv_in.load_state_dict(conv_in_sd_new)

print('out projection (conv_out) (conv_norm_out)')
out_channels = 6
norm_num_groups = 32
block_out_channels = [320]
norm_eps = 1e-5
act_fn = 'silu'
conv_out_kernel = 3
conv_out_padding = (conv_out_kernel - 1) // 2
conv_norm_out = nn.GroupNorm(
    num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
)
# uses torch.functional in orig
# conv_act = get_activation(act_fn)
conv_out = nn.Conv2d(
    block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
)

conv_norm_out.load_state_dict(model.output.gn.state_dict())
conv_out.load_state_dict(model.output.f.state_dict())

print('CONVERT')

conv_in.to('cuda')

block_one.to('cuda')
block_two.to('cuda')
block_three.to('cuda')
block_four.to('cuda')

mid_block_one.to('cuda')

up_block_one.to('cuda')
up_block_two.to('cuda')
up_block_three.to('cuda')
up_block_four.to('cuda')

conv_norm_out.to('cuda')
conv_out.to('cuda')

model.embed_image = conv_in

model.down[0] = block_one
model.down[1] = block_two
model.down[2] = block_three
model.down[3] = block_four

model.mid = mid_block_one

model.up[-1] = up_block_one
model.up[-2] = up_block_two
model.up[-3] = up_block_three
model.up[-4] = up_block_four

model.output.gn = conv_norm_out
model.output.f = conv_out

model.converted = True

sample_consistency_new = decoder_consistency(latent)
save_image(sample_consistency_new, "con_new.png")

assert (sample_consistency_orig == sample_consistency_new).all()