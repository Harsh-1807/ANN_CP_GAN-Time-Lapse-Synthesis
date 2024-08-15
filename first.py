import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from multiprocessing import freeze_support
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the StyleGAN Generator
class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim, num_frames):
        super(StyleGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_frames * 3 * 128 * 128),  # Reduced output size
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), self.num_frames, 3, 128, 128)  # Adjusted view

# Define the StyleGAN Discriminator
class StyleGANDiscriminator(nn.Module):
    def __init__(self, num_frames):
        super(StyleGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 128 * 128 * num_frames, 1024),  # Adjusted input size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

def save_sample_outputs(generator, epoch, latent_dim, num_frames, device):
    with torch.no_grad():
        sample_latent = torch.randn(1, latent_dim, device=device)
        sample_output = generator(sample_latent)
        sample_output = sample_output.cpu().numpy()
        
        for i in range(num_frames):
            frame = sample_output[0, i]
            frame = (frame * 0.5 + 0.5) * 255  # Denormalize
            frame = frame.transpose(1, 2, 0).astype('uint8')
            img = Image.fromarray(frame)
            if epoch % 100 == 0:
                img.save(f'sample_output_epoch_{epoch}_frame_{i}.png')

def train_gan(dataloader, num_epochs, latent_dim, num_frames, device, start_epoch=0):
    generator = StyleGANGenerator(latent_dim, num_frames).to(device)
    discriminator = StyleGANDiscriminator(num_frames).to(device)

    # Load previous checkpoints if available
    if start_epoch > 0:
        generator.load_state_dict(torch.load(f'generator_epoch_{start_epoch - 1}.pth'))
        discriminator.load_state_dict(torch.load(f'discriminator_epoch_{start_epoch - 1}.pth'))

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []

    plt.ion()  # Turn on interactive mode for real-time plotting

    for epoch in range(start_epoch, num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        for i, sequences in enumerate(dataloader):
            batch_size = sequences.size(0)
            
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            optimizer_d.zero_grad()

            real_sequences = sequences.to(device)
            real_loss = criterion(discriminator(real_sequences), valid)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_sequences = generator(z)
            fake_loss = criterion(discriminator(fake_sequences.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            g_loss = criterion(discriminator(fake_sequences), valid)
            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            if i % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch {i}/{len(dataloader)} '
                      f'Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}')

        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # Save models every 100 epochs
        if epoch % 100 == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')

        # Save sample outputs after each epoch
        save_sample_outputs(generator, epoch, latent_dim, num_frames, device)

        torch.cuda.empty_cache()
        gc.collect()

    return generator, discriminator

class AMOSDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = self._get_sequences()

    def _get_sequences(self):
        sequences = []
        print(f"Looking for sequences in: {self.root_dir}")
        for location_folder in os.listdir(self.root_dir):
            location_path = os.path.join(self.root_dir, location_folder)
            if os.path.isdir(location_path):
                image_files = sorted([f for f in os.listdir(location_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
                if len(image_files) >= self.sequence_length:
                    sequences.append((location_path, image_files))
        print(f"Total sequences found: {len(sequences)}")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        folder_path, image_files = self.sequences[idx]
        sequence = []
        attempts = 0
        max_attempts = 20

        while len(sequence) < self.sequence_length and attempts < max_attempts:
            start_idx = random.randint(0, len(image_files) - self.sequence_length)
            sequence = []
            for i in range(start_idx, start_idx + self.sequence_length):
                img_path = os.path.join(folder_path, image_files[i])
                try:
                    with Image.open(img_path) as img:
                        image = img.convert('RGB')
                        if self.transform:
                            image = self.transform(image)
                        sequence.append(image)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
                    break
            attempts += 1

        if len(sequence) < self.sequence_length:
            print(f"Warning: Could not find a valid sequence for {folder_path}")
            blank_image = torch.zeros(3, 128, 128)
            sequence = [blank_image] * self.sequence_length

        return torch.stack(sequence)

def main():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    amos_path = r'C:\coding\ANN_CP\datasets\AMOS'

    amos_dataset = AMOSDataset(root_dir=amos_path, sequence_length=5, transform=transform)
    dataloader = DataLoader(amos_dataset, batch_size=2, shuffle=True, num_workers=0)

    latent_dim = 100
    num_frames = 5
    num_epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine starting epoch
    start_epoch = 201
    checkpoint_files = [f for f in os.listdir() if f.startswith('generator_epoch_') and f.endswith('.pth')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        start_epoch = int(latest_checkpoint.split('_')[2].split('.')[0]) + 1
        print(f"Resuming from epoch {start_epoch}")

    try:
        freeze_support()
        generator, discriminator = train_gan(dataloader, num_epochs, latent_dim, num_frames, device, start_epoch)
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model states.")
        torch.save(generator.state_dict(), f'generator_epoch_{start_epoch - 1}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{start_epoch - 1}.pth')

if __name__ == '__main__':
    main()
