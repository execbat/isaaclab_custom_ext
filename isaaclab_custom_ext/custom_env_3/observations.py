import torch.nn.functional as F
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg

def depth_avgpool(env, sensor_cfg: SceneEntityCfg, data_type="distance_to_image_plane", pool=4, normalize=True):
    img = mdp.image(env=env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize)  # (B,H,W,1)
    img = img.permute(0,3,1,2)                       # -> (B,1,H,W)
    img = F.avg_pool2d(img, kernel_size=pool, stride=pool)
    img = F.avg_pool2d(img, kernel_size=pool, stride=pool)
    img = F.avg_pool2d(img, kernel_size=pool, stride=pool)
    return img.flatten(1)                             # -> (B, (H/p)*(W/p))
    
    
    
class compressed_image_features(mdp.image_features):
    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference (compact outputs by default)."""
        from transformers import AutoModel
        import torch
        import torch.nn.functional as F

        def _load_model() -> torch.nn.Module:
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor, *,
                       pool: str = "mean",           # "mean" | "cls" | "adaptive"
                       out_hw: tuple[int, int] | None = None,
                       flatten: bool = True) -> torch.Tensor:
            """
            Args:
                pool: "mean" → (B,192); "cls" → (B,192);
                      "adaptive" → (B,192,h,w) or (B,192*h*w) with flatten=True.
                out_hw: Objective (h,w) for adapted pool by patch map (required for pool="adaptive").
                flatten: whether to flatten (B,192,h,w) into (B,192*h*w) for pool="adaptive".
            """
            # NHWC uint8 -> NCHW float; normalization as for ImageNet
            x = images.to(model_device).permute(0, 3, 1, 2).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            x = (x - mean) / std

            # ViT: getting tokens (B, 1+N, 192), где N = (H/16)*(W/16)
            out = model.backbone.model(pixel_values=x, interpolate_pos_encoding=True)
            tokens = out.last_hidden_state   # (B, 1+N, 192)
            cls = tokens[:, 0]               # (B,192)
            patches = tokens[:, 1:]          # (B, N,192)

            if pool == "cls":
                return cls
            elif pool == "mean":
                return patches.mean(dim=1)    # (B,192)
            elif pool == "adaptive":
                assert out_hw is not None and len(out_hw) == 2, "Specify out_hw=(h,w) for pool='adaptive'."
                B, N, D = patches.shape
                # Let's restore the patch grid: (B,192, Hp, Wp)
                HpWp = int(round((N) ** 0.5))
                # If the entrance is not square, we will calculate Hp, Wp from the original dimensions
                Hp = images.shape[1] // 16
                Wp = images.shape[2] // 16
                if Hp * Wp != N:  # fallback to square approximation
                    Hp, Wp = HpWp, HpWp
                fmap = patches.reshape(B, Hp, Wp, D).permute(0, 3, 1, 2)  # (B,192,Hp,Wp)
                fmap = F.adaptive_avg_pool2d(fmap, out_hw)                 # (B,192,h,w)
                return fmap.flatten(1) if flatten else fmap
            else:
                raise ValueError(f"Unknown pool mode: {pool}")

        return {"model": _load_model, "inference": _inference}
