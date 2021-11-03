import pandas as pd
import os
import numpy as np
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
#from keras.callbacks import ModelCheckpoint
#from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_curve, auc
from scipy import interpolate
from itertools import cycle
from tensorflow.python.keras.callbacks import ModelCheckpoint


# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


n_classes = 4

train_path = "./training2017/"

# Train_data
train_files = [f for f in os.listdir(train_path) if (
    os.path.isfile(os.path.join(train_path, f)) and f[0] == 'A')]
train_bats = [f for f in train_files if f[7] == 'm']
# Choice of only 9k time steps
train_mats = [f for f in train_bats if (
    np.shape(sio.loadmat(train_path + f)['val'])[1] >= 9000)]
train_check = np.shape(sio.loadmat(train_path + train_mats[0])['val'])[1]
X = np.zeros((len(train_mats), train_check))

# Keep samples with 9000 length to achieve better class balance
for i in range(len(train_mats)):
    X[i, :] = sio.loadmat(train_path + train_mats[i])['val'][0, :9000]

# Transformation from literals (Noisy, Arithm, Other, Normal)
target_train = np.zeros((len(train_mats), 1))
# print(target_train)
Train_data = pd.read_csv(train_path + 'REFERENCE.csv',
                         sep=',', header=None, names=None)

for i in range(len(train_mats)):
    if Train_data.loc[Train_data[0] == train_mats[i][:6], 1].values == 'N':
        target_train[i] = 0
    elif Train_data.loc[Train_data[0] == train_mats[i][:6], 1].values == 'A':
        target_train[i] = 1
    elif Train_data.loc[Train_data[0] == train_mats[i][:6], 1].values == 'O':
        target_train[i] = 2
    else:
        target_train[i] = 3

# One hot encoding
Label_set_train = np.zeros((len(train_mats), n_classes))
for i in range(np.shape(target_train)[0]):
    dummy = np.zeros((n_classes))
    dummy[int(target_train[i])] = 1
    Label_set_train[i, :] = dummy

train_len = 0.8  # Choice of training size
X_train = X[:int(train_len * len(train_mats)), :]
Y_train = Label_set_train[:int(train_len * len(train_mats)), :]
X_val = X[int(train_len * len(train_mats)):, :]
Y_val = Label_set_train[int(train_len * len(train_mats)):, :]

n = 20  # seconds lasting
m = 450
c = 1  # number of channels

X_train = np.reshape(X_train, (X_train.shape[0], n, m, c))
X_val = np.reshape(X_val, (X_val.shape[0], n, m, c))

print('#############################################################################################################################################################################')
print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)


image_size = (n, m, c)

# Validation_data
val_path = './validation/'


val_files = [f for f in os.listdir(val_path) if (
    os.path.isfile(os.path.join(val_path, f)) and f[0] == 'A')]

val_bats = [f for f in val_files if f[7] == 'm']
val_mats = [f for f in val_bats if (
    np.shape(sio.loadmat(val_path + f)['val'])[1] >= 9000)]

# for f in val_bats:
#     print(np.shape(sio.loadmat(val_path + f)['val']))

val_check = np.shape(sio.loadmat(val_path + val_mats[0])['val'])[1]
X_test = np.zeros((len(val_mats), val_check))

for i in range(len(val_mats)):
    X_test[i, :] = sio.loadmat(val_path + val_mats[i])['val'][0, :9000]

val_data = pd.read_csv(val_path + 'REFERENCE.csv',
                       sep=',', header=None, names=None)

target_val = np.zeros((len(val_mats), 1))
for i in range(len(val_mats)):
    if val_data.loc[val_data[0] == val_mats[i][:6], 1].values == 'N':
        target_val[i] = 0
    elif val_data.loc[val_data[0] == val_mats[i][:6], 1].values == 'A':
        target_val[i] = 1
    elif val_data.loc[val_data[0] == val_mats[i][:6], 1].values == 'O':
        target_val[i] = 2
    else:
        target_val[i] = 3

# One hot encoding
Label_set_val = np.zeros((len(val_mats), n_classes))
for i in range(np.shape(target_val)[0]):
    dummy = np.zeros((n_classes))
    dummy[int(target_val[i])] = 1
    Label_set_val[i, :] = dummy

X_test = X_test[:(len(val_mats)), :]
X_test = np.reshape(X_test, (X_test.shape[0], n, m, c))
Y_test = Label_set_val[:(len(val_mats)):, :]

print(X_test.shape, Y_test.shape)
# print(X_val, Y_val)


# Initialising the CNN
batch_size = 32
model = tf.keras.Sequential()

# Step 1 - συνελικτικό επίπεδο 8 φίλτρων μήκους 10 με ενεργοποίηση Re.L.U.
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=10,
                                 activation='relu', input_shape=image_size, padding='same'))

# Step 2 - επίπεδο υποδειγματοληψίας τύπου “μεγίστου” (max pooling) με λόγο υποδ. 3:1
model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 1), strides=(3, 1)))

# Step 3 - συνελικτικό επίπεδο 16 φίλτρων μήκους 10 με ενεργοποίηση Re.L.U.
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=10, activation='relu', padding='same'))

# Step 4 - επίπεδο υποδειγματοληψίας τύπου “μεγίστου” (max pooling) με λόγο υποδ. 4:1
model.add(tf.keras.layers.MaxPool2D(pool_size=(4, 1)))

# Extra Step - συνελικτικό επίπεδο 16 φίλτρων μήκους 10 με ενεργοποίηση Re.L.U.
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=10, activation='relu', padding='same'))

# Extra Step - επίπεδο υποδειγματοληψίας τύπου “μεγίστου” (max pooling) με λόγο υποδ. 4:1
model.add(tf.keras.layers.MaxPool2D(pool_size=(1, 2)))

# Step 5 - Flattening
model.add(tf.keras.layers.Flatten())

# Step 6 - επίπεδο 50 πλήρως συνδεδεμένων νευρώνων με ενεργοποίηση Re.L.U.
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

# # Extra Step - Dropout
model.add(tf.keras.layers.Dropout(0.1))

# Extra Step - πλήρες συνδεδεμένος συνελικτικό επίπεδο X νευρώνων -- Working
model.add(tf.keras.layers.Dense(64, activation='relu'))

# Step 7 - Output Layer
model.add(tf.keras.layers.Dense(units=4, activation='softmax'))



# Defining the optimizer and testing it for different values
lr = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

# validation_data=(X_val, Y_val))
filepath = "saved-model-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
    save_best_only=False, mode='auto', period=1)

# Training the CNN on the Training set and evaluating it on the Test set
# validation_data=(X_val, Y_val))

b_size = 16
num_epochs = 20
model_= model.fit(X_train, Y_train, batch_size=b_size,
          epochs=num_epochs, validation_data=(X_val, Y_val), callbacks=[checkpoint])

predictions = model.predict(X_test)


def change(x):
    answer = np.zeros((np.shape(x)[0]))
    for i in range(np.shape(x)[0]):
        max_value = max(x[i, :])
        max_index = list(x[i, :]).index(max_value)
        answer[i] = max_index
    return answer.astype(np.int)


score = accuracy_score(change(Y_test), change(predictions))
# print('Learning_rate: {}'.format(optimizer.learning_rate))
print('Accuracy: {}%'.format(np.round(score, 2) * 100))


plt.figure(0)
plt.plot(model_.history['acc'], 'r')
plt.plot(model_.history['val_acc'], 'g')
plt.xticks(np.arange(0, num_epochs, 5.0))
plt.rcParams['figure.figsize'] = (10, 8)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train_acc', 'validation_acc'], loc="lower right")

plt.figure(1)
plt.plot(model_.history['loss'], 'r')
plt.plot(model_.history['val_loss'], 'g')
plt.xticks(np.arange(0, num_epochs, 5.0))
plt.rcParams['figure.figsize'] = (10, 8)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train_loss', 'validation_loss'], loc="upper right")


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(2)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
lw = 2

colors = cycle(['mediumslateblue', 'darkorange', 'cornflowerblue', 'aqua'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
