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
     
    # Method to 
    @staticmethod
    def get_device():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_cuda_present = True if torch.cuda.is_available() else False
        num_workers = 8 if is_cuda_present else 0

        return device, is_cuda_present, num_workers
    
    @staticmethod
    def showImageLab(image_Lab):
        #image_bgr = cv2.imread(image_file)
        #image_Lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

        L = image_Lab[:, :, 0]
        a = image_Lab[:, :, 1]
        b = image_Lab[:, :, 2]
        print(L)

        plt.subplot(1, 3, 1)
        plt.title('L')
        plt.gray()
        plt.imshow(L)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('a')
        plt.gray()
        plt.imshow(a)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('b')
        plt.gray()
        plt.imshow(b)
        plt.axis('off')

        plt.show() 

        

      
