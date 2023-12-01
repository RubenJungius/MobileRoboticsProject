import matplotlib.pyplot as plt
import re

def direction(p, q, r):
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

def areCollinearAndOverlapping(a1, b1, a2, b2):
    # Check if the line segments are collinear
    if direction(a1, b1, a2) == 0:
        # Check if the line segments overlap
        if a2[0] <= max(a1[0], b1[0]) and a2[0] >= min(a1[0], b1[0]) and a2[1] <= max(a1[1], b1[1]) and a2[1] >= min(a1[1], b1[1]):
            return True
    return False

def isintersect(a1, b1, a2, b2):
    # Compute the directions of the four line segments
    d1 = direction(a1, b1, a2)
    d2 = direction(a1, b1, b2)
    d3 = direction(a2, b2, a1)
    d4 = direction(a2, b2, b1)

    # Check if the two line segments intersect
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Check if the line segments are collinear and overlapping
    if areCollinearAndOverlapping(a1, b1, a2, b2) or areCollinearAndOverlapping(a2, b2, a1, b1):
        return True
    return False





def orientation(p, q, r):
    """
    Function to find the orientation of triplet (p, q, r).
    Returns:
     0 : Collinear points
     1 : Clockwise points
    -1 : Counterclockwise points
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else -1  # Clockwise or Counterclockwise

def on_segment(p, q, r):
    """
    Function to check if point q lies on line segment 'pr'.
    """
    return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and \
           q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])

def do_intersect(p1, q1, p2, q2):
    """
    Function to check if two line segments intersect.
    """

    if p1 == p2 or p1 == q2 or q1 == p2 or q1 == q2:
        return False

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases

    # p1, q1, and p2 are collinear and p2 lies on the segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return False

    # p1, q1, and q2 are collinear and q2 lies on the segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return False

    # p2, q2, and p1 are collinear and p1 lies on the segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return False

    # p2, q2, and q1 are collinear and q1 lies on the segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return False

    return False  # Doesn't fall in any of the above cases






def plot_line_segment(A, B):
    x_values = [A[0], B[0]]
    y_values = [A[1], B[1]]

    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.scatter(x_values, y_values, color='r', label='Endpoints')


A = [160, 700]
B = [746, 529]
C = [746, 529]
D = [612, 432]

   

print(do_intersect(A,B,C,D))
plot_line_segment(A,B)
plot_line_segment(C,D)

plt.title('Line Segment Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
