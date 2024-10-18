import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
from model import HS_TasNet  # Assuming HS_TasNet is defined in a file called 'main.py'
import get_wav_datasets from wav

# Custom loss function
def multi_domain_loss(final_output, target):
    # Since final_output is the combined result, use MSE loss on the output and target
    return F.mse_loss(final_output, target)

# Example of arguments setup
class Args:
    root = '/Users/samuelminkov/Desktop/Hybrid-spectogram Tasnet/dataset'
    metadata = '/path/to/metadata'
    sources = ['drums', 'bass', 'vocals', 'others']  # Define your sources
    segment = 5.0  # 5-second segments
    shift = 2.5    # 2.5-second shift between segments
    samplerate = 44100  # Desired sample rate
    channels = 2  # Stereo
    normalize = True  # Enable normalization
    full_cv = False

args = Args()

# Creating train and validation datasets
train_set, valid_set = get_wav_datasets(args)

audio_files = ['path_to_mixture_1.wav', 'path_to_mixture_2.wav']  # Mixtures
source_files = [
    ['path_to_drums_1.wav', 'path_to_bass_1.wav', 'path_to_vocals_1.wav', 'path_to_others_1.wav'],  # Sources for mixture 1
    ['path_to_drums_2.wav', 'path_to_bass_2.wav', 'path_to_vocals_2.wav', 'path_to_others_2.wav']   # Sources for mixture 2
]

# Create the dataset
dataset = AudioDataset(audio_files, source_files)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define number of epochs and learning rate
num_epochs = 10
learning_rate = 1e-4

# Define the HS-TasNet model
model = HS_TasNet(use_summation=False)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Example training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass: Get model output
        final_output = model(x)

        # Calculate loss between the final output and target (y)
        loss = multi_domain_loss(final_output, y)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate loss for reporting
        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

print("Training complete.")