# Feature parity test: firmware stage2_features.c vs Gigi compute_mel_spectrogram (verbatim copy).
# Usage (host, needs gcc + numpy + scipy), in this folder:
#   gcc -O2 -o feat driver.c stage2_features.c -lm
#   python3 compare.py
# Pass: "overall r" and "frame-level r" about 1.0 at every gain, speech-vs-quiet contrast matching in sign and size.
import numpy as np, subprocess, sys
from scipy import signal

# --- Gigi's compute_mel_spectrogram, copied verbatim from stage2/train_stage2.py ---
def compute_mel_spectrogram(audio_pcm, sr=16000, n_mels=16, n_frames=39):
    if len(audio_pcm) < sr: audio_pcm = np.pad(audio_pcm, (0, sr - len(audio_pcm)))
    else: audio_pcm = audio_pcm[:sr]
    f, t, Zxx = signal.stft(audio_pcm, fs=sr, nperseg=512, noverlap=512-400, boundary=None)
    mag = np.abs(Zxx)
    n_freqs = mag.shape[0]
    mel_filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    bin_edges = np.linspace(2, n_freqs - 5, n_mels + 2, dtype=int)
    for m in range(n_mels):
        left, center, right = bin_edges[m], bin_edges[m+1], bin_edges[m+2]
        if center > left: mel_filters[m, left:center] = np.linspace(0, 1, center - left)
        if right > center: mel_filters[m, center:right] = np.linspace(1, 0, right - center)
    mel_energy = np.dot(mel_filters, mag)
    log_mel = np.log(mel_energy + 1e-6)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    if log_mel.shape[1] < n_frames: log_mel = np.pad(log_mel, ((0, 0), (0, n_frames - log_mel.shape[1])))
    else: log_mel = log_mel[:, :n_frames]
    return log_mel.T.astype(np.float32)

def c_features(pcm_i16):
    pcm_i16.astype('<i2').tofile("in.bin")
    subprocess.run(["./feat.exe", "in.bin", "out.bin"], check=True)
    return np.fromfile("out.bin", dtype='<f4').reshape(39, 16)

rng = np.random.default_rng(0)
t = np.arange(16000) / 16000.0

def speechlike(gain):
    # silence (noise) / voiced harmonic burst with moving formant / silence (noise)
    x = 0.01 * rng.standard_normal(16000)
    seg = (t >= 0.35) & (t < 0.65)
    f0 = 140 + 40 * (t - 0.35)
    ph = 2 * np.pi * np.cumsum(f0) / 16000.0
    voiced = sum((1.0 / h) * np.sin(h * ph) for h in range(1, 25))
    env = np.sin(np.pi * np.clip((t - 0.35) / 0.30, 0, 1)) ** 2
    x = x + seg * env * voiced * 0.25
    x = x / np.max(np.abs(x))
    return np.clip(x * gain * 32767, -32768, 32767).astype(np.int16)

def corr(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

print(f"{'gain':>6} | {'overall r':>9} | {'frame-level r':>13} | {'speech-vs-quiet contrast (mean z)':>34}")
print(f"{'':>6} | {'':>9} | {'':>13} | {'python':>10} {'C':>10}")
for gain in (0.02, 0.05, 0.1, 0.25, 0.5, 0.9):
    pcm = speechlike(gain)
    py = compute_mel_spectrogram(pcm.astype(np.float32) / 32768.0)
    cc = c_features(pcm)
    fr_py, fr_c = py.mean(axis=1), cc.mean(axis=1)           # per-frame energy contour
    sp = slice(11, 24); q = np.r_[0:8, 32:39]                 # frames in word vs quiet
    con_py = fr_py[sp].mean() - fr_py[q].mean()
    con_c  = fr_c[sp].mean() - fr_c[q].mean()
    print(f"{gain:6.2f} | {corr(py, cc):9.3f} | {corr(fr_py, fr_c):13.3f} | {con_py:10.2f} {con_c:10.2f}")
