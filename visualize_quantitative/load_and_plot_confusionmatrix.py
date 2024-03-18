from utils.ConfusionMatrix import ConfusionMatrix
import pickle
import matplotlib.pyplot as plt

with open(
    "runs/kth/x3d_m__16x5____pretrained__True____train_backbone__True/0000_2023-09-06_17-38-15/confusionmatrix/data_labels.pkl",
    "rb",
) as f:
    labels = pickle.load(f)
with open(
    "runs/kth/x3d_m__16x5____pretrained__True____train_backbone__True/0000_2023-09-06_17-38-15/confusionmatrix/data_test_confusion_00079.pkl",
    "rb",
) as f:
    cm = pickle.load(f)


M = ConfusionMatrix(len(labels), labels)
M.mat = cm

a = M.as_img(fontsize=10, label_angle=90, display_values=False)
plt.imshow(a.permute(1, 2, 0))
plt.show()
