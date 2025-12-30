import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

df = pd.read_excel(
    r'C:\Users\Nicmar\Documents\coding\NER_MANUAL.xlsx'
)


y_true = [1] * len(df['ner_manual'])
y_pred = [
    1 if m == p else 0
    for m, p in zip(df['ner_manual'], df['ner_nusa_bert'])
]


acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(acc, prec, rec, f1)


cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

plt.matshow(cm, cmap='viridis')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.xticks([0, 1], ['Benar', 'Salah'])
plt.yticks([0, 1], ['Benar', 'Salah'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')


plt.show()
