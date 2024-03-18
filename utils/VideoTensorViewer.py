from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

class VideoTensorViewer(object):
    def __init__(self, vol, wrap_around=True, figsize=(8,6)):
        # vol has to be shape C,T, H, W
        self.vol = vol
        self.wrap_around = wrap_around
        self.slices = vol.shape[1]
        self.ind = 0
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.im = self.ax.imshow(self.vol[:, self.ind, ...].permute(1,2,0))
        self.update()
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def onscroll(self, event):
        if event.button == 'up':
            if not self.wrap_around and self.ind+1 == self.slices:
                return
            self.ind = (self.ind + 1) % self.slices
        else:
            if not self.wrap_around and self.ind == 0:
                return
            self.ind = (self.ind - 1) % self.slices
        plt.title(f'{self.ind}')
        self.update()

    def update(self):
        self.im.set_data(self.vol[:, self.ind, :, :].permute(1,2,0))
        self.im.axes.figure.canvas.draw()