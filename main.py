import torch
import numpy as np
import matplotlib.pyplot as plt
from  cubic_gda import cubic_gda
from gda import gda
from momentum_gda import gda_momentum

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':

    loss1, acc1, robust1 = cubic_gda(gamma = 2.0, batch_size = 512, num_epochs = 75, num_ascent_epochs = 20, lr_1 = 0.1, lr_2 = 0.01, lr_3 = 0.1, lr_4=0.001)
    loss2, acc2, robust2 = gda(gamma = 2.0, batch_size = 512, num_epochs = 75, num_ascent_epochs = 20, lr_1 = 0.05, lr_2 = 0.03)
    loss3, acc3, robust3 = gda_momentum(gamma = 2.0, beta1 = 0.5, beta2 = 0.5, batch_size = 512, num_epochs = 100, num_ascent_epochs = 10, lr_1 = 0.1, lr_2 = 0.03)

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    length = range(len(acc))

    ax1.plot(length, loss1, color="blue", lw=4, ls="-", label="cubic gda")
    ax1.plot(length, loss2, color="red", lw=4, ls="--", label="gda")
    ax1.plot(length, loss3, color="green", lw=4, ls="--", label="gda momentum")
    ax1.set_xlabel("# of epochs", fontweight="bold", fontsize=20)
    ax1.set_ylabel(r" estimate loss $ \varphi(x) $", fontweight="bold", fontsize=20)
    ax1.legend(fontsize=16)

    ax2.plot(length, robust1, color="blue", lw=4, ls="-")
    ax2.plot(length, robust2, color="red", lw=4, ls="--")
    ax2.plot(length, robust3, color="green", lw=4, ls="--")
    ax2.set_xlabel("# of epochs", fontweight="bold", fontsize=20)
    ax2.set_ylabel("robust test accuracy", fontweight="bold", fontsize=20)
    ax2.legend(fontsize=16, loc="upper left", labels=["cubic gda", "gda", "gda momentum"])

    plt.show()

