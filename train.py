import os
import configparser
import torch
import torch.nn as nn
import numpy as np
import rasterio
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from dataset.multi_channel_dataset import UAV5ChannelDataset

# =====================================================================
# 1. 5-CHANNEL MASK R-CNN MODEL BUILDER
# =====================================================================

def get_5channel_maskrcnn(num_classes):
    """
    Builds a Mask R-CNN model adapted for 5-channel inputs:
    Channels 0-2: RGB
    Channel 3: NDVI
    Channel 4: CHM
    """
    # 1. Load a pre-trained ResNet-50 FPN backbone
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=torchvision.models.ResNet50_Weights.DEFAULT,
        trainable_layers=3
    )
    
    # 2. Modify the first Conv2d layer (conv1) to take 5 channels instead of 3
    old_conv = backbone.body.conv1
    new_conv = nn.Conv2d(
        in_channels=5,  # [R, G, B, NDVI, CHM]
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias
    )
    
    # 3. Transfer pre-trained weights & warm-start NDVI/CHM channels
    with torch.no_grad():
        # Copy RGB weights directly
        new_conv.weight[:, :3] = old_conv.weight
        # Initialize NDVI and CHM channels with the mean of RGB weights
        mean_rgb_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:] = mean_rgb_weight.repeat(1, 2, 1, 1)
        
    backbone.body.conv1 = new_conv

    # 4. Initialize Mask R-CNN with modified backbone
    model = MaskRCNN(backbone, num_classes=num_classes)

    # 5. Replace Box Predictor Head for custom class count
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # 6. Replace Mask Predictor Head for custom class count
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def collate_fn(batch):
    return tuple(zip(*batch))

# =====================================================================
# 3. TRAINING & EXECUTION BOILERPLATE
# =====================================================================

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    config.read(config_path)

    if not config.has_section("TRAIN"):
        raise ValueError(f"Missing [TRAIN] section in {config_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Set up class counts (Background=0, Tree=1)
    NUM_CLASSES = int(config.get("TRAIN", "num_classes", fallback="2"))

    # Initialize Model
    model = get_5channel_maskrcnn(num_classes=NUM_CLASSES)
    model.to(device)

    # Example setup for Dataset paths
    rgb_files = ["tile_0_rgb.tif", "tile_1_rgb.tif"]
    ndvi_files = ["tile_0_ndvi.tif", "tile_1_ndvi.tif"]
    chm_files = ["tile_0_chm.tif", "tile_1_chm.tif"]

    # Placeholder target structure per tile
    dummy_annotations = [{
        'boxes': [[50.0, 50.0, 200.0, 200.0]],               # [xmin, ymin, xmax, ymax]
        'labels': [1],                                       # Target Class
        'masks': np.zeros((1, 640, 640), dtype=np.uint8)     # Binary mask [N, H, W]
    }] * len(rgb_files)

    # DataLoader
    batch_size = int(config.get("TRAIN", "batch_size", fallback="2"))
    dataset = UAV5ChannelDataset(rgb_files, ndvi_files, chm_files, dummy_annotations)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataset = UAV5ChannelDataset(rgb_files, ndvi_files, chm_files, dummy_annotations)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer and training configuration
    learning_rate = float(config.get("TRAIN", "learning_rate", fallback="1e-4"))
    weight_decay = float(config.get("TRAIN", "weight_decay", fallback="1e-4"))
    num_epochs = int(config.get("TRAIN", "num_epochs", fallback="10"))
    checkpoint_dir = config.get("TRAIN", "checkpoint_dir", fallback="checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(data_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {batch_idx}/{len(data_loader)} Loss: {losses.item():.4f}")

        train_loss = running_loss / max(1, len(data_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {train_loss:.4f}")

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, (val_images, val_targets) in enumerate(val_loader, 1):
                val_images = [img.to(device) for img in val_images]
                val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

                val_loss_dict = model(val_images, val_targets)
                val_losses = sum(loss for loss in val_loss_dict.values())
                val_running_loss += val_losses.item()

                print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Batch {val_batch_idx}/{len(val_loader)} Loss: {val_losses.item():.4f}")

        val_loss = val_running_loss / max(1, len(val_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Validation Loss: {val_loss:.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"maskrcnn_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
        }, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")