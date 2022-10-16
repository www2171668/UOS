""""""
import numpy as np
import math
def point_distance_line(point, line_point1, line_point2):
    # 计算向量
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance

def isPointInsideSegment(point, line_point1, line_point2):
    # 求Cos∠PP1P2
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2
    dx10 = x0 - x1;
    dy10 = y0 - y1;
    m10 = math.hypot(dx10, dy10);
    dx12 = x2 - x1;
    dy12 = y2 - y1;
    m12 = math.hypot(dx12, dy12);
    if ((dx10 * dx12 + dy10 * dy12) / m10 / m12 < 0): return False

    # 求Cos∠PP2P1
    dx20 = x0 - x2;
    dy20 = y0 - y2;
    m20 = math.hypot(dx20, dy20);
    dx21 = x1 - x2;
    dy21 = y1 - y2;
    m21 = math.hypot(dx21, dy21);
    return (dx20 * dx21 + dy20 * dy21) / m20 / m21 >= 0

point = np.array([5, 2])
line_point1 = np.array([2, 2])
line_point2 = np.array([3, 3])

print(isPointInsideSegment(point, line_point1, line_point2))
print(point_distance_line(point, line_point1, line_point2))
