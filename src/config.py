SEED = 42

# Audio
SR = 32000                          # sampling rate after resampling (Hz)
DURATION_S = 5                      # every clip is exactly 5 seconds
N_SAMPLES = SR * DURATION_S         # 160000 samples per clip

# Spectrogram
N_FFT = 1024                        # window length used for each Fourier transform
WIN_LENGTH = 1024
HOP = 320                           # step between windows (10 ms at 32 kHz)
N_MELS = 64                         # number of mel bands
FMIN = 50
FMAX = 14000
N_FRAMES = 1 + N_SAMPLES // HOP     # 501 columns in the spectrogram

# Dataset
N_CLASSES = 50
N_FOLDS = 5

# Numerics
EPS = 1e-8                          # avoids division by zero