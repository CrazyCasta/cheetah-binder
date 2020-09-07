import numpy

# TODO: Write a ton of unit tests for this
# dx1, dy1 must be counter clockwise of dx2, dy2. dx1, dx2 both >= 0. If not,
# flip vector.
#     -
#    / \
#   /a d\
#  o     |
#   \b c/
#    \ /
#     -
# a => y <= mx + b => dx1 * y - dy1 * x <= 0
# b => dx2 * y - dy2 * x >= 0
# c => dx1 * (y-dy2) - dy1 * (x-dx2) >= 0
# d => dx2 * (y-dy1) - dy2 * (x-dx1) <= 0
def parallelogram(dx1, dy1, dx2, dy2, data, center=[1,1]):
    count_a = max(data, key=lambda x: len(x))
    count_b = len(data)
    size_x = count_a * math.ceil(dx1) + count_b * math.ceil(dx2)
    size_y = count_a * math.ceil(dy1) + count_b * math.ceil(dy2)

    min_x = min(0, count_a * dx1) + min(0, count_b * dx2)
    min_y = min(0, count_a * dy1) + min(0, count_b * dy2)

    result = numpy.zeros((size_x, size_y))

    for j, data_line in enumerate(data):
        for i, data_bit in enumerate(data_line):
            # TODO: This could be improved with some bounding boxes
            for k in range(size_x):
                for l in range(size_y):
                    x = k - min_x - i * dx1 - j * dx2
                    y = l - min_y - i * dy1 - j * dy2
                    if (dx1 * y - dy1 * x) <= 0 and \
                       (dx2 * y - dy2 * x) >= 0 and \
                       (dx1 * (y-dy2) - dy1 * (x-dx2)) >= 0 and \
                       (dx2 * (y-dy1) - dy2 * (x-dx1)) <= 0:
                        result[k, l] = data_bit

    return result
