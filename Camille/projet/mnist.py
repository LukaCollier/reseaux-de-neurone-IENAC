import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px
import Activation
import Neurone


# Charger le CSV
data = pd.read_csv("mnist.csv")

# Séparer labels et images
X = data.iloc[:, 1:].values  # pixels
y = data.iloc[:, 0].values   # labels
#normaliser
X = X / 255.0

def one_hot_encode(y, n_classes=10):
    return np.eye(n_classes)[y]

y_encoded = one_hot_encode(y)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


relu = Activation.ActivationF.relu()
softmax = Activation.ActivationF.softmax()

# Création du réseau
nn = Neurone.Neural_Network(
    n_input_init=784,
    nb_n_l=[128, 10],
    activ=[relu, softmax],
    loss="cross_entropy"
)

#entrainement:
nn.train_ADAM(
    x_train=X_train,
    y_train=y_train,
    epochs=50,
    lr=0.001,
    batch_size=256,
    x_val=X_val,
    y_val=y_val
)

y_pred = nn.forward(X_val)
y_pred_labels = np.argmax(y_pred, axis=0)  # y_pred est (10, n_samples)
y_true_labels = np.argmax(y_val, axis=1)   # y_val est (n_samples, 10)

accuracy = np.mean(y_pred_labels == y_true_labels)

df = pd.DataFrame({
    "epoch": list(range(len(nn.train_losses))),
    "train_loss": nn.train_losses,
    "val_loss": nn.val_losses
})

df_long = df.melt(id_vars="epoch", value_vars=["train_loss", "val_loss"],
                  var_name="type", value_name="loss")

fig = px.line(df_long, x="epoch", y="loss", color="type",
              title="Évolution de la loss train / validation")
fig.show()
