# Evaluating Dataset Level Bias Mitigation in CNN Melanoma Classification
Evaluating Dataset-Level Bias Mitigation Strategies for Subgroup Fairness in CNN-Based Melanoma Classification Across Fitzpatrick Skin Types
## Description
- Trained a binary melanoma classifier on the Fitzpatrick17k and ISIC Archive datasets and evaluated three bias mitigation strategies: dataset filtering, adversarial training (FGSM), and weighted diversification, across Fitzpatrick skin tone subgroups
- Computed fairness metrics including Equal Opportunity Difference, Demographic Parity Difference, and Disparate Impact Ratio, finding that no strategy reduced the recall disparity between lighter and darker skin tone groups despite strong aggregate AUC (0.97–0.995)
- Demonstrated that high overall model accuracy can mask complete subgroup failure, with one condition achieving DIR = 0.00 on darker-skin patients despite AUC > 0.97
## Created using
- Python
- PyTorch
- EfficientNet-B0
