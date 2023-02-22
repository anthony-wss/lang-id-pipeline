import whisper
import torchaudio
import librosa
import torch
import numpy as np

def batched_log_mel_spectrogram(audio):
    mel_spec = torch.tensor(librosa.feature.melspectrogram(y=np.array(audio.cpu()), sr=16000, n_fft=400, hop_length=160, n_mels=80))

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

transform = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=80, norm='slaney', mel_scale="slaney")

audio, sr = torchaudio.load("EasySpanish/zyLoe3k_LaI.ogg")
# audio = torch.stack([audio.squeeze(), audio.squeeze(), audio.squeeze()]) #.squeeze()
audio = whisper.pad_or_trim(audio)
mel = batched_log_mel_spectrogram(audio)[:, :-1]
# mel2 = whisper.log_mel_spectrogram(audio)
mel3 = transform(audio)[:, :-1]
log_spec = torch.clamp(mel3, min=1e-10).log10()
log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
mel3 = (log_spec + 4.0) / 4.0

print(audio.shape)
print(mel.shape)
# print(mel2.shape)
print(mel3.shape)

print(mel[0][:40])
print(mel3[0][:40])

mel_numpy = np.array(mel)
# mel2_numpy = np.array(mel2)
mel3_numpy = np.array(mel3)
dist_squared = np.sum(np.square(mel_numpy - mel3_numpy))
print(dist_squared)

exit()

tokens, probs = self.whisper_model.detect_language(mel[:,:,:-1]) # Pop the last column