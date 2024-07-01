import torch
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from dataset import SRDataset, TestSRDataset
from model import RRDBNet
from PIL import Image
import os
from config import test_lr_dir,output_hr_dir

model_path="/home/rishabh.mondal/Brick-Kilns-project/albk_rishabh/albk_v2/YOLO_LOCALIZATION/ESRGAN/models/RRDB_PSNR_x4.pth"

test_dataset = TestSRDataset(lr_dir=test_lr_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)

model.load_state_dict(torch.load(model_path, map_location='cpu'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


# Testing loop
with torch.no_grad():
    for lr_img, filename in test_loader:
        lr_img = lr_img.to(device)
        
        # print(lr_img.shape)
        # Forward pass
        sr_img = model(lr_img)

        # print(sr_img.shape)

        # input()


        # Resize the SR output to desired size if needed (e.g., using F.interpolate)
        sr_img = torch.nn.functional.interpolate(sr_img, size=(1120, 1120), mode='bilinear', align_corners=False)
        
        # print(sr_img.shape)

        # input()

        # Convert tensor to PIL image
        sr_img = sr_img.squeeze(0).cpu()
        sr_img = to_pil_image(sr_img)

        # Create the output file name based on the original filename
        output_subdir = os.path.join(output_hr_dir, os.path.dirname(filename[0]))

        # output_subdir = os.path.join(output_hr_dir, os.path.splitext(filename[0])[0])
        os.makedirs(output_subdir, exist_ok=True)

        # Save the super-resolved image
        output_path = os.path.join(output_subdir, filename[0])
        sr_img.save(output_path)

        print(f'Saved: {output_path}')