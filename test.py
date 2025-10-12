import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import preprocessing
from diffusers import UNet1DModel, DDPMScheduler, DDIMScheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error
from scipy.fftpack import fft

def get_rms(records, multi_channels):
    if multi_channels == 1:
        n = records.shape[0]
        rms = 0
        for i in range(n):
            rms_t = np.sum([records[i]**2]/len(records[i]))
            rms += rms_t
        return rms/n
    
    if multi_channels == 0:
        rms = np.sum([records**2])/ len(records)
        return rms

def snr(signal, noisy):
    snr = 10 * np.log10(signal/noisy)
    return snr

def random_signal(signal, comb):
    res = []

    for i in range(comb):
        rand_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[rand_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
        res.append(shuffled_dataset)
    
    random_result = np.array(res)

    return random_result

def prepare_data(comb):
    eeg_data = np.load('./data/EEG_all_epochs.npy')
    noise_data = np.load('./data/EMG_all_epochs.npy')

    eeg_random = np.squeeze(random_signal(signal=eeg_data, comb=1))
    noise_random = np.squeeze(random_signal(signal=noise_data, comb=1))

    reuse_num = noise_random.shape[0] - eeg_random.shape[0]
    eeg_reuse = eeg_random[0: reuse_num, :]
    eeg_random = np.vstack([eeg_reuse, eeg_random])
    print(f'EEG shape after crop and resuse to match EMG samples: {eeg_random.shape[0]}')

    t = noise_random.shape[1]
    train_num = round(eeg_random.shape[0] * 0.9)
    test_num = round(eeg_random.shape[0] - train_num)

    train_eeg = eeg_random[0: train_num, :]
    test_eeg = eeg_random[train_num: train_num + test_num,:]

    train_noise = noise_random[0: train_num, :]
    test_noise = noise_random[train_num: train_num+test_num, :]

    EEG_train = random_signal(signal=train_eeg, comb=comb).reshape(comb * train_eeg.shape[0],t)
    NOISE_train = random_signal(signal=train_noise, comb=comb).reshape(comb * train_noise.shape[0], t)

    EEG_test = random_signal(signal=test_eeg, comb=comb).reshape(comb * test_eeg.shape[0],t)
    NOISE_test = random_signal(signal=test_noise, comb=comb).reshape(comb * test_noise.shape[0], t)

    print(f"train data clean shape: {EEG_train.shape}")
    print(f"train data noise shape: {NOISE_train.shape}")

    sn_train = []
    eeg_train = []
    all_sn_test = []
    all_eeg_test = []

    SNR_train_dB = np.random.uniform(-5.0, 5.0, (EEG_train.shape[0]))
    print(SNR_train_dB.shape)
    SNR_train = np.sqrt(10**(0.1*(SNR_train_dB)))


    for i in range(EEG_train.shape[0]):
        noise = preprocessing.scale(NOISE_train[i])
        EEG = preprocessing.scale(EEG_train[i])

        alpha = get_rms(EEG, 0) / (get_rms(noise, 0 ) * SNR_train[i])
        noise *= alpha
        signal_noise = EEG + noise

        sn_train.append(signal_noise)
        eeg_train.append(EEG)
    
    SNR_test_dB = np.linspace(-5.0, 5.0, num=(11))
    SNR_test = np.sqrt(10 ** (0.1 * SNR_test_dB))

    for i in range(11):
        sn_test = []
        eeg_test = []
        for k in range(EEG_test.shape[0]):
            noise = preprocessing.scale(NOISE_test[k])
            EEG = preprocessing.scale(EEG_test[k])

            alpha = get_rms(EEG,0) / (get_rms(noise, 0) * SNR_test[i])
            noise *= alpha
            signal_noise = EEG + noise

            sn_test.append(signal_noise)
            eeg_test.append(EEG)
        
        sn_test = np.array(sn_test)
        eeg_test = np.array(eeg_test)

        all_sn_test.append(sn_test)
        all_eeg_test.append(eeg_test)
    
    X_train = np.array(sn_train)
    y_train = np.array(eeg_train)

    X_test = np.array(all_sn_test)
    y_test = np.array(all_eeg_test)

    X_train = np.expand_dims(X_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    return [X_train, y_train, X_test, y_test]

X_train, y_train, X_test, y_test = prepare_data(11)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
print(type(X_train), type(y_train))
X_train.to(device), y_train.to(device)

model = UNet1DModel(
    sample_size=512,
    in_channels=2,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64,128,256),
    down_block_types=(
        "DownBlock1D",
        "AttnDownBlock1D",
        "AttnDownBlock1D",
    ),
    up_block_types=(
        "AttnUpBlock1D",
        "AttnUpBlock1D",
        "UpBlock1D",
    ),
)
model.to(device)
model.parameters

train_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
inference_scheduler = DDIMScheduler(num_train_timesteps=1000,  beta_schedule="squaredcos_cap_v2")
optim = torch.optim.AdamW(model.parameters(), lr=0.001)

def forward_process(clean, mix):
    noise = torch.randn_like(clean)
    batch_size = clean.size(0)
    t = torch.randint(0, train_scheduler.num_train_timesteps, (batch_size,)).long()
    t.to(device)
    x_t = train_scheduler.add_noise(clean, noise, t)
    model_in = torch.cat([x_t, mix], dim=1)
    eps_pred = model(model_in, t).sample
    loss = F.mse_loss(eps_pred, noise)
    
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    return loss.item()

def train_epoch(dataloader, epoch):
    model.train()
    loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (X_noisy, y_clean) in enumerate(progress_bar):
        X_noisy = X_noisy.to(device)
        y_clean = y_clean.to(device)

        cur_loss = forward_process(y_clean, X_noisy)
        loss += cur_loss

        progress_bar.set_postfix({'loss': f'{loss:.4f}'})
    
    avg_loss = loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def reverse_process(mix, steps=50):
    model.eval()
    x = torch.randn_like(mix)
    inference_scheduler.set_timesteps(steps)
    for t in inference_scheduler.timesteps:
        model_in = torch.cat([x, mix], dim=1)
        eps = model(model_in, t).sample
        x = inference_scheduler.step(eps, t, x).prev_sample
    
    return x

from torch.utils.data import TensorDataset

def train_model(X_train, y_train, n=100, batch_size=32):
    dataset = TensorDataset(y_train, X_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    best_loss = float('inf')
    for e in range(n):
        avg_loss = train_epoch(train_loader, e+1)
        print(f"epoch {e+1}/{n} | Average Loss {avg_loss}")
    
    return model

def SNR(y1, y2):
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2-y1), axis=1)
    SNR = 10 * np.log10(N/D)
    return np.mean(SNR)

def SNR_improvement(y_in, y_out, y_clean):
    return SNR(y_clean, y_out) - SNR(y_clean - y_in)

def RRMSE(denoised, clean):
    denoised, clean = denoised.squeeze(), clean.squeeze()
    rmse1 = np.sqrt(mean_squared_error(denoised, clean))
    rmse2 = np.sqrt(mean_squared_error(clean, np.zeros(clean.shape, dtype=float)))

    return rmse1/rmse2

def get_PSD(records):
    x_fft = fft(records, 400)
    x_fft = np.abs(x_fft)
    psd = x_fft ** 2 / 400
    return psd

def RRMSE_s(denoise, clean):
    clean = clean.squeeze()
    denoise = denoise.squeeze()
    rmse1 = np.sqrt(mean_squared_error(get_PSD(denoise), get_PSD(clean)))
    rmse2 = np.sqrt(mean_squared_error(get_PSD(clean), np.zeros(clean.shape, dtype=float)))

    return rmse1/rmse2

@torch.no_grad()
def eval_model(X_test, y_test, num_inference_steps=50):
    res = {}

    for s in range(X_test.shape[0]):
        snr_db = -5 + s
        print(f"Evaluating SNR noise level: {snr_db}")

        X_test_snr = X_test[s]
        y_test_snr = y_test[s]

        X_test_tensor = torch.FloatTensor(X_test_snr).to(device)

        batch_size = 32
        y_pred_list = []

        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            y_pred_batch = reverse_process(batch, num_inference_steps)
            y_pred_list.append(y_pred_batch.cpu().numpy())
        
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        y_clean_2d = y_test_snr.squeeze(1)
        y_noisy_2d = X_test_snr.squeeze(1)
        y_pred_2d = y_pred.squeeze(1)

        SNR_improvement = SNR_improvement(y_noisy_2d, y_pred_2d, y_clean_2d)
        rrmse_temp = np.mean(RRMSE(y_pred[i], y_test_snr[i]) for i in range(len(y_pred)))
        rrmse_spect = np.mean(RRMSE_s(y_pred[i], y_test_snr[i]) for i in range(len(y_pred)))
        CC = np.corrcoef(y_test_snr.flatten(), y_pred.flatten())[0,1]

        res[s] = {
            'snr_improvement': SNR_improvement,
            'rrmse_time': rrmse_temp,
            'rrmse_freq': rrmse_spect,
            'correlation': CC
        }
        print(f"  SNR Improvement: {SNR_improvement:.2f} dB")
        print(f"  Temporal RRMSE: {rrmse_temp:.4f}")
        print(f"  Spectral RRMSE: {rrmse_spect:.4f}")
        print(f"  Correlation Coeff: {CC:.4f}")

model = train_model(
    X_train=X_train,
    y_train=y_train,
    n=100,
    batch_size=32
    )