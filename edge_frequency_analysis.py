import math

import matplotlib.pyplot
from PIL import Image
import numpy
import scipy.signal


# Assuming data dimensions are odd, if not then blame the user
def imshow_shifted(axes, data):
    xmin = -data.shape[0] // 2
    xmax = data.shape[0] + xmin - 1
    ymin = -data.shape[1] // 2
    ymax = data.shape[1] + ymin - 1
    axes.imshow(data, extent=(xmin, xmax, ymin, ymax), origin='lower')


class EdgeFrequencyAnalysis:
    figsize = (8,17)
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
            figure, axeses = matplotlib.pyplot.subplots(2, 1,
                                                        sharex=True,
                                                        sharey=True,
                                                        figsize=self.figsize)

            axeses[0].imshow(self.barcode)
            axeses[1].imshow(self.edges)

            figure.show()

    def analyze(self):
        self.edges_f = numpy.fft.fft2(self.edges)
        display_data = numpy.fft.fftshift(abs(self.edges_f) /
                                          self.edges_f.size)

        figure, axeses = matplotlib.pyplot.subplots(2, 1,
                                                    sharex=True, sharey=True,
                                                    figsize=self.figsize)

        imshow_shifted(axeses[0], display_data)
        imshow_shifted(axeses[1], display_data**0.5)

        figure.show()

    def analyze_subzones(self, x_zones, y_zones):
        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)

        axes.imshow(self.edges)
        for i in range(128,512,128):
            axes.axvline(i)
            axes.axhline(i)

        figure.show()


        x_step = self.edges.shape[0] // x_zones
        y_step = self.edges.shape[0] // y_zones
        self.sub_edge_f = numpy.zeros((x_step,y_step))

        for i in range(x_zones):
            for j in range(y_zones):
                part = self.edges[x_step*i:x_step*(i+1),
                                  y_step*j:y_step*(j+1)]
                part -= sum(sum(part)) / part.size
                self.sub_edge_f += abs(numpy.fft.fft2(part))


        sub_grid = numpy.meshgrid(range(-x_step//2,x_step//2), range(-y_step//2,y_step//2))
        x_grid, y_grid = sub_grid
        scale_factor = (x_grid**2.0+y_grid**2.0)**0.5


        figure = matplotlib.pyplot.figure()

        axes = figure.add_subplot(1,1,1)
        imshow_shifted(axes, numpy.fft.fftshift(self.sub_edge_f)*scale_factor)
        #axes.set_xlim(0,50)
        #axes.set_ylim(0,50)

        figure.show()


        figure = matplotlib.pyplot.figure()

        axes = figure.add_subplot(1,1,1)
        imshow_shifted(axes, numpy.fft.fftshift(self.sub_edge_f))
        #axes.set_xlim(0,50)
        #axes.set_ylim(0,50)

        figure.show()


    def analyze_slopes(self, resolution=60):
        # Making the assumption this is square, if not...
        x_size, y_size = self.edges_f.shape
        abs_edges_f = abs(self.edges_f) / x_size / y_size

        # Kill the noise near DC
        abs_edges_f[0:4,0:4] *= 0
        abs_edges_f[-3:,0:4] *= 0
        abs_edges_f[0:4,-3:] *= 0
        abs_edges_f[-3:,-3:] *= 0

        slopes = [_/resolution for _ in range(-resolution,resolution)]
        y_results = [] # y = slope * x
        x_results = [] # x = slope * y

        for slope in slopes:
            y_result = 0
            x_result = 0
            for i in range(x_size//2):
                for j in range(-1, 2):
                    a, b = i, (round(slope*i)+j)%x_size
                    if a > x_size // 2:
                        a -= x_size
                    if b > x_size // 2:
                        b -= x_size
                    y_result += abs_edges_f[a, b]
                    x_result += abs_edges_f[b, a]
            y_results.append(y_result)
            x_results.append(x_result)


        figure, axeses = matplotlib.pyplot.subplots(1,1)
        axeses.plot(slopes, y_results)
        axeses.plot(slopes, x_results)
        figure.show()


    def iter_maximums(self):
        edges_f_copy = abs(numpy.array(self.edges_f))
        x_size = edges_f_copy.shape[0]
        y_size = edges_f_copy.shape[0]

        # Note: this is not the best way to iterate through most of the
        # locations, but the expectation is that the user is only interested in
        # a small fraction of locations 
        for i in range(edges_f_copy.size):
            max_loc = edges_f_copy[:x_size,:y_size].argmax()

            x_max_loc, y_max_loc = max_loc%(x_size), max_loc//(x_size)
            x_max_loc_print = x_max_loc
            y_max_loc_print = y_max_loc

            if x_max_loc_print > x_size // 2:
                x_max_loc_print -= x_size
            if y_max_loc_print > y_size // 2:
                y_max_loc_print -= y_size

            # Don't yield the aliases
            if x_max_loc + y_max_loc <= (x_size + y_size) // 2:
                yield x_max_loc_print, y_max_loc_print

            edges_f_copy[y_max_loc, x_max_loc] = -math.inf


    def list_maximums(self, count, ignore_square_lt=4):
        x_step = self.edges_f.shape[0]
        y_step = self.edges_f.shape[1]
        edges_f_copy = scipy.signal.convolve2d(abs(self.edges_f),
                                               numpy.ones((3,3)),
                                               'same', 'symm')
        edges_f_copy = numpy.array(self.edges_f)

        if False:
            figure = matplotlib.pyplot.figure()

            axes = figure.add_subplot(1,1,1)
            axes.imshow(abs(edges_f_copy[:self.edges.shape[0],
                                         :self.edges.shape[1]]))

            figure.show()

        results = []

        for i in range(count):
            max_loc = abs(edges_f_copy[:x_step,:y_step]).argmax()
            x_max_loc, y_max_loc = max_loc%(x_step), max_loc//(x_step)
            x_max_loc_print = x_max_loc
            y_max_loc_print = y_max_loc
            if x_max_loc_print > x_step // 2:
                x_max_loc_print -= x_step
            if y_max_loc_print > y_step // 2:
                y_max_loc_print -= y_step

            # Don't print the aliases
            if (x_max_loc + y_max_loc <= (x_step + y_step) // 2 and
                (abs(x_max_loc_print) >= ignore_square_lt or
                 abs(y_max_loc_print) >= ignore_square_lt)):
                value = abs(edges_f_copy[y_max_loc, x_max_loc])/x_step/y_step
                metric_a = value
                metric_b = (abs(edges_f_copy[x_max_loc, -y_max_loc])
                            /x_step/y_step)
                metric_c = (abs(edges_f_copy[2*y_max_loc_print+1,
                                             2*x_max_loc_print+1])
                            /x_step/y_step)
                metric_d = (abs(edges_f_copy[2*x_max_loc_print+1,
                                             2*-y_max_loc_print+1])
                            /x_step/y_step)
                metric = metric_a + 2*metric_b + 2*metric_c + 3*metric_d
                print(f"{x_max_loc_print:5d} {y_max_loc_print:5d} "
                      f"{value:6.3f} {metric}")
                results.append((x_max_loc_print, y_max_loc_print,
                                value, metric))
            edges_f_copy[y_max_loc, x_max_loc] = 0

        print("Sorted by metric")
        sorted_results = sorted(results, key=lambda x: x[3])
        for x_max_loc_print, y_max_loc_print, value, metric in sorted_results:
            print(f"{x_max_loc_print:5d} {y_max_loc_print:5d} "
                  f"{value:6.3f} {metric}")

    def composite(self, x_f, y_f, amplitude=50):
        x, y = numpy.meshgrid(range(self.edges.shape[0]),
                              range(self.edges.shape[1]))
        self.ideal = numpy.sin(x_f*2*math.pi/self.edges.shape[0] * x +
                               y_f*2*math.pi/self.edges.shape[1] * y)
        self.edges_composite = self.edges + amplitude * self.ideal

        figure = matplotlib.pyplot.figure()
        axes = figure.add_subplot(1,1,1)

        axes.imshow(self.edges_composite)

        figure.show()
