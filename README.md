# Humankind-Memory-Inspired-Few-Shot-Class-Incremental-learning
## Abstract
Current mainstream deep learning techniques exhibit an over-reliance on extensive training data and a lack of adaptability to the dynamic world, marking a considerable disparity from human intelligence. To bridge this gap, _Few-Shot Class-Incremental Learning_ (FSCIL) has emerged, focusing on continuous learning of new categories with limited samples without forgetting old knowledge. Existing FSCIL studies typically use a single model to learn knowledge across all sessions, inevitably leading to the stability-plasticity dilemma. Unlike machines, humans store varied knowledge in different cerebral cortices. Inspired from this characteristic, our paper aims to develop the method which learns independent models for each session. It can inherently prevent catastrophic forgetting. During the testing phrase, our method integrates _Uncertainty Quantification_ (UQ) for model deployment. Our method provides a fresh viewpoint for FSCIL and demonstrates the state-of-the-art performance on CIFAR-100 and _mini_-ImageNet datasets.
## Methodology
<div style="text-align:center">
  <img src="Fig/Introduction.png">
</div>
Comparing ordinary incremental learning method to parameter-isolation method.


<div style="text-align:center">
  <img src="Fig/IJICAI.png">
</div>
The overview of our proposed humankind memory-inspired FSCIL approach.
