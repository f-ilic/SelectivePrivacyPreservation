from matplotlib import pyplot as plt


class SliceViewer(object):
    # vol has to be shape T, H, W, C
    def __init__(
        self, vol, captions=None, cmap="gray", wrap_around=True, figsize=(16, 4)
    ):
        self.vol = vol
        self.captions = captions
        self.wrap_around = wrap_around
        self.slices = vol.shape[0]
        self.ind = 0

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)
        self.im = self.ax.imshow(self.vol[self.ind, ...], cmap=cmap)
        self.update()
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def onscroll(self, event):
        if event.button == "up":
            if not self.wrap_around and self.ind + 1 == self.slices:
                return
            self.ind = (self.ind + 1) % self.slices
        else:
            if not self.wrap_around and self.ind == 0:
                return
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.vol[self.ind, ...])
        title_string = f"SCROLL UP/DOWN: {self.ind}/{self.slices - 1}"
        self.ax.set_title(title_string)
        self.im.axes.figure.canvas.draw()
