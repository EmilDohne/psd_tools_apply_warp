import numba
from numba.experimental import jitclass


@jitclass([
    ('x', numba.float64),
    ('y', numba.float64),
])
class Point2d:

    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    @staticmethod
    def lerp(p1, p2, t: float):
        """Linear interpolation between two points."""
        return Point2d(
            p1.x + (p2.x - p1.x) * t,
            p1.y + (p2.y - p1.y) * t
        )


    def __len__(self):
        # A Point2D has two components (x and y)
        return 2

    def __getitem__(self, index: int):
        # Allow indexing: 0 for x, 1 for y
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Index out of range for Point2D")
    
    # Arithmetic operations with another Point2d
    def __add__(self, other):
        if isinstance(other, Point2d):
            return Point2d(self.x + other.x, self.y + other.y)
        raise TypeError(f"Unsupported operand type(s) for +: 'Point2d' and '{type(other)}'")

    def __sub__(self, other):
        if isinstance(other, Point2d):
            return Point2d(self.x - other.x, self.y - other.y)
        raise TypeError(f"Unsupported operand type(s) for -: 'Point2d' and '{type(other)}'")

    # Scalar multiplication and division
    def __mul__(self, scalar: float):
        if isinstance(scalar, (int, float)):
            return Point2d(self.x * scalar, self.y * scalar)
        raise TypeError(f"Unsupported operand type(s) for *: 'Point2d' and '{type(scalar)}'")

    def __truediv__(self, scalar: float):
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ValueError("Cannot divide by zero")
            return Point2d(self.x / scalar, self.y / scalar)
        raise TypeError(f"Unsupported operand type(s) for /: 'Point2d' and '{type(scalar)}'")

    # Arithmetic operations with a Point2d and a scalar
    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __rtruediv__(self, scalar: float):
        return self.__truediv__(scalar)