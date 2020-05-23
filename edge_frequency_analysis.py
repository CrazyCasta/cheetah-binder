import matplotlib.pyplot
from PIL import Image
import numpy
import scipy.signal

class EdgeFrequencyAnalysis:
    def __init__(self, image_filename):
        self.image = Image.open(image_filename)
        self.grey_image = self.image.convert("L")

        self.numpy_image = numpy.asarray(self.grey_image)
        self.crop(0, 0, self.numpy_image.shape[1], self.numpy_image.shape[0],
                  False)

    def crop(self, x, y, width, height, show=True):
        # 800, 800+256*5
        # 1500, 1500+256*5
        x2 = x + width
        y2 = y + width

        self.barcode = self.numpy_image[y:y2, x:x2].astype(float)
        self.edges = (numpy.pad(abs(self.barcode[1:] - self.barcode[:-1]),
                                ((0,1),(0,0))) +
                      numpy.pad(abs(self.barcode[:,1:] - self.barcode[:,:-1]),
                                ((0,0),(0,1))))

        self.edges -= sum(sum(self.edges)) / self.edges.size

        if show:
            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot(1,1,1)

            axes.imshow(self.barcode)

            figure.show()


            figure = matplotlib.pyplot.figure()
            axes = figure.add_subplot(1,1,1)

            axes.imshow(self.edges)

            figure.show()

    def analyze(self):
        self.edges_f = numpy.fft.fft2(self.edges)


        figure = matplotlib.pyplot.figure()

        axes = figure.add_subplot(1,1,1)
        axes.imshow(abs(self.edges_f[:self.edges.shape[0]//2,
                                     :self.edges.shape[1]//2]))

        figure.show()

    def list_maximums(self, count):
        x_step = self.edges_f.shape[0]
        y_step = self.edges_f.shape[1]
        edges_f_copy = scipy.signal.convolve2d(abs(self.edges_f),
                                               numpy.ones((3,3)),
                                               'same', 'symm')

        for i in range(count):
            max_loc = abs(edges_f_copy[:x_step//2,:y_step//2]).argmax()
            x_max_loc, y_max_loc = max_loc%(x_step//2), max_loc//(x_step//2)
            print(x_max_loc, y_max_loc,
                  abs(edges_f_copy[y_max_loc, x_max_loc]))
            edges_f_copy[y_max_loc, x_max_loc] = 0
