import librosa
import random
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import soundfile as sf
import math
import time
import torch
from scipy.io import savemat
import os
import scipy.io as io
import torch.nn.functional as functional
import torchaudio
from trainer.base_trainer import BaseTrainer
from util.utils import compute_STOI, compute_PESQ, overlap_cat
from util.pip import NetFeeder, Resynthesizer, wavNormalize
plt.switch_backend('agg')
import imageio
from model.CR import CR
from scipy.io import savemat

class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
    

    def _train_epoch(self, epoch):

        train_loss = []
        loss_total = 0.0
        num_batchs = len(self.train_data_loader)
        num_index = 0
        with tqdm(total = num_batchs) as pbar:
            for i, (mixture, clean, name) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()

                
                mixture = mixture.to(self.device) #[B, T]
                clean = clean.to(self.device) #[B, T]

                mixture_d = mixture(clean.type(torch.cuda.FloatTensor))                
                clean_d = feeder(clean.type(torch.cuda.FloatTensor))
				
                enhanced_d = self.model(mixture_d.type(torch.cuda.FloatTensor))

                clean_mag = clean_mag.type(torch.cuda.FloatTensor)
                enhanced = resynthesizer(enhanced_d, mixture.type(torch.cuda.FloatTensor))
                f_p1, f_p2, f_p3, f_p4, f_n1, f_n2, f_n3, f_n4, f_a1, f_a2, f_a3, f_a4 = CR(enhanced, mixture, clean)

                loss1 = self.loss_function(enhanced, clean)
                loss2 = self.loss_function(f_a1, f_p1)/self.loss_function(f_a1, f_n1) * 32
                loss3 = self.loss_function(f_a2, f_p2)/self.loss_function(f_a2, f_n2) * 16
                loss4 = self.loss_function(f_a3, f_p3)/self.loss_function(f_a3, f_n3) * 8
                loss5 = self.loss_function(f_a4, f_p4)/self.loss_function(f_a4, f_n4) * 4              
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
                num_index += 1

                pbar.update(1)
                
        end_time = time.time()
        

        dl_len = len(self.train_data_loader)
        print("loss:", loss_total / dl_len)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]
        num_batchs = len(self.validation_data_loader)
        loss_total = 0.0
        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []

        with tqdm(total = num_batchs) as pbar:
            for i, (mixture, clean, name) in enumerate(self.validation_data_loader):
                assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
                name = name[0]
                padded_length = 0
                mixture = mixture.to(self.device) #[B, T]

                clean = clean.to(self.device) #[B, T]

				
                mixture = mixture.to(self.device) #[B, T]
                clean = clean.to(self.device) #[B, T]

                mixture_d = mixture(clean.type(torch.cuda.FloatTensor))                
                clean_d = feeder(clean.type(torch.cuda.FloatTensor))
				
                enhanced_d = self.model(mixture_d.type(torch.cuda.FloatTensor))

                clean_mag = clean_mag.type(torch.cuda.FloatTensor)
                enhanced = resynthesizer(enhanced_d, mixture.type(torch.cuda.FloatTensor))
                f_p1, f_p2, f_p3, f_p4, f_n1, f_n2, f_n3, f_n4, f_a1, f_a2, f_a3, f_a4 = CR(enhanced, mixture, clean)

                loss1 = self.loss_function(enhanced, clean)
                loss2 = self.loss_function(f_a1, f_p1)/self.loss_function(f_a1, f_n1) * 32
                loss3 = self.loss_function(f_a2, f_p2)/self.loss_function(f_a2, f_n2) * 16
                loss4 = self.loss_function(f_a3, f_p3)/self.loss_function(f_a3, f_n3) * 8
                loss5 = self.loss_function(f_a4, f_p4)/self.loss_function(f_a4, f_n4) * 4              
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                loss_total += loss.item()


                mixture = mixture.detach().squeeze(0).cpu().numpy()
                clean = clean.detach().squeeze(0).cpu().numpy()
                enhanced = enhanced.detach().squeeze(0).cpu().numpy()


                assert len(mixture) == len(enhanced) == len(clean)
          
                pbar.update(1)

        print("loss:", loss_total/num_batchs)
        score = loss_total
        
        return score
