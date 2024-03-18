privacy_models = {
    "resnet18": {
        "num_frames": 1,
        "sample_rate": 1,
        "crop_size": 224,
    },
    "resnet50": {
        "num_frames": 1,
        "sample_rate": 1,
        "crop_size": 224,
    },
    "resnet101": {
        "num_frames": 1,
        "sample_rate": 1,
        "crop_size": 224,
    },
    "vit": {
        "num_frames": 1,
        "sample_rate": 1,
        "crop_size": 224,
    },
    "vit_b_32": {
        "num_frames": 1,
        "sample_rate": 1,
        "crop_size": 224,
    },
}

action_models = {
    "x3d_s": {
        "crop_size": 182,
        "num_frames": 13,
        "sample_rate": 6,
    },
    "x3d_m": {
        "crop_size": 224,
        "num_frames": 16,
        "sample_rate": 5,
    },
    "x3d_l": {
        "crop_size": 312,
        "num_frames": 16,
        "sample_rate": 5,
    },
    "slowfast_r50": {
        "crop_size": 224,
        "num_frames": 32,
        "sample_rate": 2,
        "frames_per_second": 30,
        "slowfast_alpha": 4,
    },
    "slow_r50": {
        "crop_size": 224,
        "num_frames": 8,
        "sample_rate": 8,
    },
    "i3d_r50": {
        "crop_size": 224,
        "num_frames": 8,
        "sample_rate": 8,
    },
    "c2d_r50": {
        "crop_size": 224,
        "num_frames": 8,
        "sample_rate": 8,
    },
    "csn_r101": {
        "crop_size": 224,
        "num_frames": 32,
        "sample_rate": 2,
    },
    "r2plus1d_r50": {
        "crop_size": 224,
        "num_frames": 16,
        "sample_rate": 4,
    },
    "mvit_base_16x4": {
        "crop_size": 224,
        "num_frames": 16,
        "sample_rate": 4,
    },
    "e2s_x3d_s": {
        "crop_size": 182,
        "num_frames": 13,
        "sample_rate": 6,
    },
    "e2s_x3d_m": {
        "crop_size": 224,
        "num_frames": 16,
        "sample_rate": 5,
    },
    "e2s_x3d_l": {
        "crop_size": 312,
        "num_frames": 16,
        "sample_rate": 5,
    },
}

valid_models = dict(list(privacy_models.items()) + list(action_models.items()))
