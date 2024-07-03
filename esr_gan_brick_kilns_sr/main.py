# main.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SRDataset
from model import RRDBNet
from config import lr_dir, hr_dir, val_hr, val_lr, lr_size, hr_size, batch_size, learning_rate, num_epochs

# Create the dataset and dataloader
train_dataset = SRDataset(lr_dir=lr_dir, hr_dir=hr_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = SRDataset(lr_dir=val_lr, hr_dir=val_hr)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize your model
model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
model_path = '/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/ESRGAN/models/RRDB_PSNR_x4.pth'
model.load_state_dict(torch.load(model_path), strict=True)

# Define the loss function and optimizer
criterion = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for lr_imgs, hr_imgs in train_loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        sr_imgs = model(lr_imgs)

        # Resize the SR output to match HR size if needed (e.g., using F.interpolate)
        sr_imgs = torch.nn.functional.interpolate(sr_imgs, size=hr_size, mode='bilinear', align_corners=False)

        # Compute the loss
        loss = criterion(sr_imgs, hr_imgs)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * lr_imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            sr_imgs = torch.nn.functional.interpolate(sr_imgs, size=hr_size, mode='bilinear', align_corners=False)
            loss = criterion(sr_imgs, hr_imgs)
            val_loss += loss.item() * lr_imgs.size(0)

    val_loss /= len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), '/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/super_res_exp/Time-Series-Super-Resolution/esr_gan_brick_kilns_sr/licknow_sarath_fine_tuned_esrgan_epochs_100.pth')
