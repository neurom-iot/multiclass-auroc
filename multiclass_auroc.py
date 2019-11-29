import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import csv
import sys

def binarize(rightLabel, predictLabel):
    binarizedLabel=[]
    for i, right in enumerate(rightLabel):
        if right == predictLabel[i]:
            binarizedLabel.append(1)
        else:
            binarizedLabel.append(0)
    return binarizedLabel

def csvOpen(dir):
    f = open(dir, 'r')
    rdr = csv.reader(f)
    rightLabel=[]
    predictLabel=[]
    probLabel=[]
    for line in rdr:
        rightLabel.append(line[0])
        predictLabel.append(line[1])
        probLabel.append(float(line[2]))
    f.close()
    return rightLabel, predictLabel, probLabel

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("csv path is empty.")
        sys.exit()
    rightLabel, predictLabel, probLabel = csvOpen(sys.argv[1])

    binarizedLabel = binarize(rightLabel,predictLabel)

    class_F = np.array(binarizedLabel)
    proba_F = np.array(probLabel)

    false_positive_rate_F, true_positive_rate_F, thresholds_F = roc_curve(class_F, proba_F)
    roc_auc_F = auc(false_positive_rate_F, true_positive_rate_F)

    print("AUC = ", roc_auc_F)

    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate(1 - Specificity)')
    plt.ylabel('True Positive Rate(Sensitivity)')


    plt.plot(false_positive_rate_F, true_positive_rate_F, 'b', label='Model F (AUC = %0.2f)'% roc_auc_F)
    plt.plot([0,1],[1,1],'y--')
    plt.plot([0,1],[0,1],'r--')

    plt.legend(loc='lower right')
    plt.show()