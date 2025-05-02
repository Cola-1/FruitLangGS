import argparse 
import torch
import numpy as np
import open_clip
import os
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from autoencoder.model import Autoencoder

def crop_xyz_mask(xyz, cube):
    half_cube = cube / 2.0
    mask = ((xyz[:, 0] >= -half_cube) & (xyz[:, 0] <= half_cube) &
            (xyz[:, 1] >= -half_cube) & (xyz[:, 1] <= half_cube) &
            (xyz[:, 2] >= -half_cube) & (xyz[:, 2] <= half_cube))
    return mask

def parse_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, tuple):
        data = ckpt[0]
    else:
        data = ckpt
    xyz = data[1].to(device)
    features_dc = data[2].to(device).squeeze(1).clamp(0, 1)
    features_rest = data[3].to(device)
    scales = data[4].to(device)
    rotations = data[5].to(device)
    opacity = data[6].to(device).squeeze(-1)
    semantic_features = data[7].to(device)
    return {
        'xyz': xyz,
        'features_dc': features_dc,
        'features_rest': features_rest,
        'scales': scales,
        'rotations': rotations,
        'opacity': opacity,
        'semantic_features': semantic_features,
    }

def load_autoencoder(ckpt_path, encoder_dims, decoder_dims, device):
    model = Autoencoder(encoder_dims, decoder_dims).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

def get_text_features(prompts, device):
    model_dir = os.path.join(os.path.dirname(__file__), "clip_vit_b16")
    model_path = os.path.join(model_dir, "open_clip_pytorch_model.bin")
    model = open_clip.create_model('ViT-B-16')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    text = open_clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat

def cosine_match(gauss_feats, text_feats, thresh=0.2):
    sims = torch.matmul(gauss_feats, text_feats.T)
    max_sim, _ = torch.max(sims, dim=1)
    print(f"Max similarity for positive match: {max_sim.max().item()}")
    mask = max_sim > thresh
    return mask.nonzero(as_tuple=True)[0].to(gauss_feats.device)

def cosine_filter_negative(gauss_feats, neg_text_feats, neg_thresh=0.3):
    sims = torch.matmul(gauss_feats, neg_text_feats.T)
    max_sim, _ = torch.max(sims, dim=1)
    print(f"Max similarity for negative match: {max_sim.max().item()}")
    mask = max_sim <= neg_thresh
    return mask

def save_gaussians_to_ply(xyz, scales, rots, opacities, colours, ply_path):
    xyz_np = xyz.cpu().numpy()
    s_np = scales.cpu().numpy()
    r_np = rots.cpu().numpy()
    opa_np = opacities.cpu().numpy()
    col_np = (colours.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [(f"scale_{i}", "f4") for i in range(s_np.shape[1])]
    dtype += [(f"rot_{i}", "f4") for i in range(r_np.shape[1])]
    dtype += [("opacity", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    v = np.empty(xyz_np.shape[0], dtype=dtype)
    v["x"], v["y"], v["z"] = xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2]
    for i in range(s_np.shape[1]):
        v[f"scale_{i}"] = s_np[:, i]
    for i in range(r_np.shape[1]):
        v[f"rot_{i}"] = r_np[:, i]
    v["opacity"], v["red"], v["green"], v["blue"] = opa_np, col_np[:, 0], col_np[:, 1], col_np[:, 2]
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(ply_path)
    print(f"[?] Exported PLY: {ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract Gaussians matching language prompts")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prompts", nargs='+', type=str, required=True)
    parser.add_argument("--N_prompts", nargs='+', type=str, default=[])
    parser.add_argument("--autoencoder_ckpt", type=str, required=True)
    parser.add_argument("--cosine_thresh", type=float, default=0.2)
    parser.add_argument("--neg_thresh", type=float, default=0.3)
    parser.add_argument("--cube_size", type=float, default=10.0)
    parser.add_argument('--encoder_dims', nargs='+', type=int, default=[256, 128, 64, 32, 3])
    parser.add_argument('--decoder_dims', nargs='+', type=int, default=[16, 32, 64, 128, 256, 256, 512])
    args = parser.parse_args()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    base_name = "_".join(args.prompts)
    output_ply_path = os.path.join(output_dir, f"{base_name}.ply")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[1] Loading checkpoint")
    ckpt = parse_checkpoint(args.checkpoint_path, device)
    if args.cube_size > 0:
        print("[1.1] Applying scene cropping with cube_size:", args.cube_size)
        mask = crop_xyz_mask(ckpt['xyz'], cube=args.cube_size)
        for key in ['xyz', 'features_dc', 'features_rest', 'scales', 'rotations', 'opacity', 'semantic_features']:
            ckpt[key] = ckpt[key][mask]

    print("[2] Decoding semantic features")
    model = load_autoencoder(args.autoencoder_ckpt, args.encoder_dims, args.decoder_dims, device)
    decoded = model.decode(ckpt['semantic_features'])
    decoded = decoded / decoded.norm(dim=-1, keepdim=True)

    print("[3] Encoding positive prompts")
    text_feats = get_text_features(args.prompts, device)
    matched_ids = cosine_match(decoded, text_feats, args.cosine_thresh)

    if args.N_prompts:
        print("[4] Encoding negative prompts")
        neg_text_feats = get_text_features(args.N_prompts, device)
        keep_mask = cosine_filter_negative(decoded[matched_ids], neg_text_feats, args.neg_thresh)
        matched_ids = matched_ids[keep_mask]

    print(f"[5] Selected {len(matched_ids)} Gaussians")
    if len(matched_ids) > 0:
        save_gaussians_to_ply(
            ckpt['xyz'][matched_ids],
            ckpt['scales'][matched_ids],
            ckpt['rotations'][matched_ids],
            ckpt['opacity'][matched_ids],
            ckpt['features_dc'][matched_ids],
            output_ply_path
        )
    else:
        print("[Warning] No matched Gaussians were found. Nothing exported.")
