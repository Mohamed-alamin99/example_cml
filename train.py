# train.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

X,y = load_iris(return_X_y=True)
Xtr,Xte,ytr,yte = train_test_split(X,y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(Xtr,ytr)
pred = clf.predict(Xte)

acc = accuracy_score(yte,pred)
cm = confusion_matrix(yte,pred)

# save metrics
with open("metrics.txt","w") as f:
    f.write(f"accuracy: {acc:.4f}\n")

# save a simple plot
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("plot.png")