"""MAE-pretrained-trunk pipeline: joint SCG+ECG masked-autoencoder pretraining on
unlabeled multi-domain data (teacher/pretrain_mae.py's approach), weak-supervision MIL
auto-labeling, distillation to the MCU-sized SCGNet, and the Monte Carlo evaluation
comparing MAE-pretrained vs. from-scratch trunks on real Chelten labels."""
