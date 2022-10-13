import sys
import pickle
import matplotlib.pyplot as plt


class Series:
    def render(self):
        plt.plot(self.xx, self.yy, self.style, label=self.label, color=self.color)

class Lines:
    def render(self):
        if self.vh=='v': pl = plt.axvline
        if self.vh=='h': pl = plt.axhline
        for x in self.xx:
            pl(x, linestyle=self.style, label=self.label, color=self.color)

class Figure:

    def __init__(self, xlabel, ylabel, xticks=None, title=None, size=(5, 3), tight=True, grid=True, legend=True):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xticks = xticks
        self.title = title
        self.size = size
        self.tight = tight
        self.grid = grid
        self.legend = legend
        self.ss = []

    def add_series(self, xx, yy, label='', style='', color=None):
        s = Series()
        s.xx = xx
        s.yy = yy
        s.label = label
        s.style = style
        s.color = color
        self.ss.append(s)

    def add_lines(self, vh, xx, label='', style='', color=None):
        s = Lines()
        s.xx = xx
        s.vh = vh
        s.label = label
        s.style = style
        s.color = color
        self.ss.append(s)

    def render(self):
        for s in self.ss: s.render()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if self.xticks: plt.xticks(self.xticks)
        if self.title!=None: plt.title(self.title)
        if self.grid: plt.grid()
        if self.legend: plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(self.size[0], self.size[1])
        plt.tight_layout()

    def show(self):
        self.render()
        plt.show()

    def savefig(self, path_fn):
        self.render()
        plt.savefig(path_fn)
        plt.clf()

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


if __name__=='__main__':
    path = sys.argv[1]
    with open(path, 'rb') as f:
        fig = pickle.load(f)
    fig.show()








