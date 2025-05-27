
import os

import torch
from typing_extensions import Literal

from PIL import Image
from torchvision import transforms as tvf
import glob
import natsort
from tqdm.auto import tqdm

from utilities import DinoV2ExtractFeatures

# Main
if __name__ == "__main__":

    # weights_path = "checkpoints/dinov2_vits14_pretrain.pth"
    # model = load_dinov2_with_custom_weights(weights_path)
    # print(model)

    # Program parameters
    # save_dir = _ex("./data/CityCenter/GD_Images/")
    device = torch.device("cuda")
    # Dino_v2 properties (parameters)
    desc_layer: int = 10
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    num_c: int = 32
    # Domain for use case (deployment environment)
    domain: Literal["aerial", "indoor", "urban", "UPS"] = "UPS"
    # Maximum image dimension
    max_img_size: int = 1024

    extractor = DinoV2ExtractFeatures("dinov2_vits14", desc_layer, desc_facet, device=device)

    # Base image transformations
    base_tf = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    if (True):

        # Global descriptor generation
        imgs_dir = '/home/ubuntu/Anantak/SensorUnit/data/Map/ImageData/000/00'
        assert os.path.isdir(imgs_dir), "Input directory doesn't exist!"
        img_fnames = glob.glob(f"{imgs_dir}/*.png")
        img_fnames = natsort.natsorted(img_fnames)
        # if largs.first_n is not None:
        # if (False):
        #     img_fnames = img_fnames[:10]

        print(f"Calculating descriptors for {len(img_fnames)} images")
        patch_descs = []
        for img_fname in tqdm(img_fnames):
            # DINO features
            with torch.no_grad():
                pil_img = Image.open(img_fname).convert('RGB')
                img_pt = base_tf(pil_img).to(device)
                # Make image patchable (14, 14 patches)
                c, h, w = img_pt.shape
                h_new, w_new = (h // 14) * 14, (w // 14) * 14
                img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
                # Extract descriptor
                ret = extractor(img_pt) # [1, num_patches, desc_dim]
                # print(f"    input dim = {c}-{h}-{w}->{h_new}-{w_new}, {(h // 14)*(w // 14)} descriptor dim = {ret.shape}")
                patch_descs.append(ret.cpu())
                # patch_descs.append({"img": pil_img, "descs": ret.cpu()})

        patch_descs = torch.cat(patch_descs, dim=0) # [N, n_p, d_dim]


