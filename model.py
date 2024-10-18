import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramEncoder(nn.Module):
    def __init__(self, window_size, hop_size):
        super(SpectrogramEncoder, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size

    def forward(self, x):
        # Compute the STFT and return the magnitude spectrogram
        spec = torch.stft(x, n_fft=self.window_size, hop_length=self.hop_size, return_complex=False)
        return spec

class TimeDomainEncoder(nn.Module):
    def __init__(self, num_basis, kernel_size, stride):
        super(TimeDomainEncoder, self).__init__()
        self.conv1d = nn.Conv1d(1, num_basis, kernel_size, stride=stride)

    def forward(self, x):
        # Apply learned convolution to the time domain waveform
        x = x.unsqueeze(1)  # Add channel dimension
        encoded = F.relu(self.conv1d(x))
        return encoded

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output

class MaskEstimation(nn.Module):
    def __init__(self, input_size, output_size):
        super(MaskEstimation, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Estimate the mask to apply to the encoded features
        mask = torch.sigmoid(self.fc(x))
        return mask

class SpectrogramDecoder(nn.Module):
    def __init__(self, window_size, hop_size):
        super(SpectrogramDecoder, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size

    def forward(self, x):
        recon_wave = torch.istft(x, n_fft=self.window_size, hop_length=self.hop_size)
        return recon_wave

class TimeDomainDecoder(nn.Module):
    def __init__(self, num_basis, kernel_size, stride):
        super(TimeDomainDecoder, self).__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(num_basis, 1, kernel_size, stride=stride)

    def forward(self, x):
        decoded = self.conv_transpose1d(x)
        return decoded

class HS_TasNet(nn.Module):
    def __init__(self, window_size=1024, hop_size=512, num_basis=1024, kernel_size=16, lstm_hidden_size=500, use_summation=False):
        super(HS_TasNet, self).__init__()
        self.spectrogram_encoder = SpectrogramEncoder(window_size, hop_size)
        self.time_domain_encoder = TimeDomainEncoder(num_basis, kernel_size, stride=kernel_size // 2)

        # LSTM blocks for both branches
        self.spectrogram_lstm = LSTMBlock(513, lstm_hidden_size, num_layers=2)  # Spectrogram encoder produces 513 features
        self.time_lstm = LSTMBlock(num_basis, lstm_hidden_size, num_layers=2)

        # Combined LSTM branch
        self.use_summation = use_summation  # Flag to switch between sum or concatenation

        if use_summation:
            self.combined_lstm = LSTMBlock(lstm_hidden_size, lstm_hidden_size, num_layers=1)  # Smaller version (HS-TasNet-Small)
        else:
            self.combined_lstm = LSTMBlock(2 * lstm_hidden_size, 1000, num_layers=2)  # Original version with 1000 units

        # Mask estimation layers for both branches
        self.spectrogram_mask_estimation = MaskEstimation(lstm_hidden_size, 513)
        self.time_mask_estimation = MaskEstimation(lstm_hidden_size, num_basis)

        # Decoders
        self.spectrogram_decoder = SpectrogramDecoder(window_size, hop_size)
        self.time_domain_decoder = TimeDomainDecoder(num_basis, kernel_size, stride=kernel_size // 2)

        # Skip connections
        self.skip_connection_spec = nn.Identity()  # Direct pass-through for spectrogram
        self.skip_connection_time = nn.Identity()  # Direct pass-through for time domain

        # Output layers for each source (drums, bass, vocals, others)
        self.num_sources = 4
        self.source_output_layers = nn.ModuleList([nn.Linear(lstm_hidden_size, lstm_hidden_size) for _ in range(self.num_sources)])

    def forward(self, x):
        # Encode spectrogram and time-domain
        spec_features = self.spectrogram_encoder(x)
        time_features = self.time_domain_encoder(x)

        # Apply LSTM to both encoded features
        spec_lstm_output = self.spectrogram_lstm(spec_features)
        time_lstm_output = self.time_lstm(time_features)

        # Combine LSTM outputs
        if self.use_summation:
            combined = spec_lstm_output + time_lstm_output  # Summation for HS-TasNet-Small
        else:
            combined = torch.cat([spec_lstm_output, time_lstm_output], dim=-1)  # Concatenation for original HS-TasNet

        combined_output = self.combined_lstm(combined)

        # Split the combined output back to spectrogram and time-domain branches
        if self.use_summation:
            spec_output_split = combined_output  # No need to split in the smaller version
            time_output_split = combined_output
        else:
            spec_output_split = combined_output[:, :513, :]
            time_output_split = combined_output[:, 513:, :]

         # Skip connections
        skip_spec = self.skip_connection_spec(spec_output_split)
        skip_time = self.skip_connection_time(time_output_split)

        # Mask estimation
        spec_mask = self.spectrogram_mask_estimation(skip_spec)
        time_mask = self.time_mask_estimation(skip_time)

        # Apply masks
        masked_spec = spec_features * spec_mask
        masked_time = time_features * time_mask

        # Decode
        decoded_spec = self.spectrogram_decoder(masked_spec)
        decoded_time = self.time_domain_decoder(masked_time)

        # Combine the decoded outputs from both branches
        final_output = decoded_spec + decoded_time

        sources_output = []
        for i in range(self.num_sources):
            source_output = self.source_output_layers[i](combined_decoded_output)
            sources_output.append(source_output)

        return sources_output