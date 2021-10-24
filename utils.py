import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

class Utils:
    @staticmethod
    def get_ab_mean(a_channel, b_channel):
        a_channel_mean = a_channel.mean(dim=(2, 3))
        b_channel_mean = b_channel.mean(dim=(2, 3))
        a_b_mean = torch.cat([a_channel_mean,
                              b_channel_mean], dim=1)
        return a_b_mean

    @staticmethod
    def plot_loss_epoch(train_loss_avg, fig_name):
        plt.ion()
        fig = plt.figure()
        plt.plot(train_loss_avg)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.show()
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()