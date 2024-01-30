# 1. Introduction
- Attribute: A meaningful feature inherent in an image such as hair color, gender or age.
- Attribute value: A particular value of an attribute, e.g., black/blond/brown for hair color or male/female for gender.
- Domain: A set of images sharing the same attribute value.
- Multi-domain image-to-image translation: We change images according to attributes from multiple domains.
- Training multiple domains from different datasets is possible, such as jointly training CelebA and RaFD images to change a CelebA image’s facial expression using features learned by training on RaFD.
- 현존 모델들의 문제점:
    - $k$개의 Domain이 있을 때 $_{k}P_{2}$개의 모델이 필요합니다.
    - 전체 Domain에 대한 공통된 Features가 있다고 하더라고 2개의 Domain으로부터밖에 학습이 불가합니다.
- 모델은 Domain의 정보를 One-hot encoded label로서 받아들입니다. 학습 중에는 Target domain label을 무작위로 정하고 그 Domain으로 이미지를 번역하도록 모델이 학습됩니다.
- Adding a mask vector to the domain label:
    - 모르는 Label을 무시하고 특정 데이터셋에 의해 주어지는 Label에만 집중하도록 합니다.

# 2. Related Work
- 생략합니다.

# 3. Star Generative Adversarial Networks
## 3.1) MultiDomain Image-to-Image Translation
- $x$: Source domain image.
- $y$: Target domain image.
- $c$: (Randomly generated) target domain label.
- $c'$: Source domain.
$$G(x, c) \rightarrow y$$
- $D_{\text{src}}(x)$: Probability distribution over sources.