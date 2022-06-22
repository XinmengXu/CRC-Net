# CRC-Net
"All Information is Necessary: Exploring the Interactions between Speech Positive and Negative Information for Monaural Speech Enhancement"

Monaural speech enhancement (SE) is a challenging ill-posed problem due to the irreversible degradation process. Most supervised learning-based methods solve this ill-posed problem by calculating the loss function between estimates samples and ground truth and by adopting local processing to allocate great attention to more speech relevant regions of a feature map. However, these methods achieve SE tasks that rely solely on positive information, e.g., ground-truth speech and speech relevant feature regions. Different from above, we observe that the negative information, such as noisy speech and speech irrelevant feature regions, are valuable to guide the SE model training process. Therefore, we propose a SE model encoder-decoder architecture with the guidance of negative information, which consists of two innovations, (1) we propose a contrastive regularization (CR) built upon contrastive learning to ensure that the estimated speech is pulled closer to the clean speech and pushed far away from the noisy speech in the representation space, and (2) we design a collaboration module, which contains three parts, speech-relevant block, speech-irrelevant block, and interactive block, to establish the correlation between the speech relevant and irrelevant information at the global-level and local-level in a learnable and self-adaptive manner. We term the proposed SE network with contrastive regularization and collaboration module as CRC-Net. Experimental results demonstrate that our CRM-Net achieves more comparable and superior performance than recent approaches.

Submitted to AAAI 2023.