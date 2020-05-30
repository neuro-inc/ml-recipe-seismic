from src.data_types import Point


def projection(p1, p2, p):
    """Find the projection of a point p on the line [p1, p2]"""
    A = (p1.y - p2.y) / (p1.x - p2.x)
    B = p1.y - A * p1.x
    x = ((p.y - B) * A + p.x) / (1 + A ** 2 )
    y = A * x + B
    return Point(x, y)


if __name__ == '__main__':
    pass
