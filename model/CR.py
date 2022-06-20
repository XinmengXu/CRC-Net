import os
import soundfile as sf
import IPython
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio
import torchaudio.models as m


def CR(p, n, a)
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
	
    with torch.inference_mode():
        features_p, _ = model.extract_features(p)
        features_n, _ = model.extract_features(n)
        features_a, _ = model.extract_features(a)

    f_p1 = features_p[1] + features_p[2] + features_p[3]
    f_p2 = features_p[4] + features_p[5] + features_p[6]
    f_p3 = features_p[7] + features_p[8] + features_p[9]
    f_p4 = features_p[10] + features_p[11] + features_p[12]

    f_n1 = features_n[1] + features_n[2] + features_n[3]
    f_n2 = features_n[4] + features_n[5] + features_n[6]
    f_n3 = features_n[7] + features_n[8] + features_n[9]
    f_n4 = features_n[10] + features_n[11] + features_n[12]
	
    f_a1 = features_a[1] + features_a[2] + features_a[3]
    f_a2 = features_a[4] + features_a[5] + features_a[6]
    f_a3 = features_a[7] + features_a[8] + features_a[9]
    f_a4 = features_a[10] + features_a[11] + features_a[12]
	
return f_p1, f_p2, f_p3, f_n1, f_n2, f_n3, f_a1, f_a2, f_a3