import sys

import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open("text_model_info.txt", 'r', encoding="utf-8") as f:
        info = f.readlines()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for line in info:
        split = line.split("  ")
        
        tl, ta, vl, va = split[1], split[2], split[4], split[5]
        tl = tl.split(": ")[-1]
        ta = ta.split(": ")[-1]
        vl = vl.split(": ")[-1]
        va = va.split(": ")[-1][:-2]

        train_losses.append(float(tl))
        val_losses.append(float(vl))
        train_accs.append(float(ta))
        val_accs.append(float(va))


    epochs = list(range(0, 300))
    

    # Plotting validation and test accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accs, label='Validation Accuracy', c='r')
    plt.plot(epochs, train_accs, label='Train Accuracy',  c='b')
    plt.title('Train/Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.grid(True)
    plt.savefig('text_model_acc.png')

    plt.clf()

    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_losses, label='Validation Loss', c='r')
    plt.plot(epochs, train_losses, label='Train Loss', c='b')
    plt.title('Train/Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.grid(True)
    plt.savefig('text_model_loss.png')
        