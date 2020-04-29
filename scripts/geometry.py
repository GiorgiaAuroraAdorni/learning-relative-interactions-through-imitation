import numpy as np
from copy import deepcopy


class Geometry:
    """
    Abstract class that represents a hierarchy of geometric entities and
    provides methods to process them.

    Geometric entities should inherit this class to mark their participation in
    the hierarchy. In addition, they should provide all the operations that make
    sense on them specifically.
    """

    def children(self):
        """
        Returns the direct children of this geometric entity.

        The default implementation returns all instance attributes, as provided
        by the Python standard method `vars(self)`. Subclasses can override this
        method to use any other policy to define children.

        :return: dict containing the children and their keys
        """
        return vars(self)

    def descendants(self, cls=None, prefix=()):
        """
        Returns all descendants of this geometric entity by traversing the
        hierarchy in depth-first order.

        :param cls: return only descendants of this type (default: `Geometry`)
        :param prefix: the prefix used during recursion to form the fully-qualified keys

        :return: dict containing the descendants and their fully-qualified keys
        """

        if cls is None:
            cls = Geometry

        out = dict()

        for key, value in self.children().items():
            fq_key = (*prefix, key)

            if isinstance(value, cls):
                out[fq_key] = value

            if isinstance(value, Geometry):
                out.update(value.descendants(cls, fq_key))

        return out

    def transformed(self, H):
        """
        Returns a deep copy of this geometric entity, transformed by the
        supplied transformation, H.

        At the moment only homographies, represented as 3⨉3 matrices, are supported.

        The default implementation simply makes a deep copy of `self` and
        applies `transformed(H)` recursively to all of its children.

        Subclasses that represent "concrete" entities (as opposed to simply
        being collections of other entities) should override this method to also
        apply the transformation on themselves.

        :param H: a geometric transformation
        :return: the entity resulting from the transformation
        """
        copy = deepcopy(self)

        for key, child in copy.children().items():
            if isinstance(child, Geometry):
                setattr(copy, key, child.transformed(H))

        return copy


class Vector(np.ndarray, Geometry):
    """
    Abstract class for geometric entities represented as vectors or matrices.

    Subclass of `numpy.ndarray` that also inherits from `Geometry` to participate
    in the hierarchy and to add a convenience constructor that behaves like
    `numpy.asarray(...)`.
    """

    def __new__(cls, *args, **kwargs):
        """
        Create an instance of the specified class, initialized with the content
        passed in the first parameter.

        The behavior of this constructor has three differences compared to `numpy.asarray(...)`:
          1. The returned array is always an instance of the class it has been called on, `cls`.
          2. It does not try to infer the dtype from the data, it's always `float64` unless a
             different one is explicitly specified in kwargs.
          3. Calling the constructor with a single `None` parameter, results in None being
             returned (instead of an array of dtype `object` with a single None element).
        """

        if len(args) == 1 and args[0] is None:
            return None

        kwargs.setdefault("dtype", np.float64)

        return np.asarray(*args, **kwargs).view(cls)


class Point(Vector):
    """One or more points expressed in homogeneous coordinates. Expected shape: 3⨉n."""

    @staticmethod
    def from_list(points):
        """
        Create a Point object from the most natural way of expressing it,
        converting it to the right shape.

        :param points: iterable of points. Expected shape: n⨉3
        :return: a Point object with the correct shape
        """
        return Point(points).T

    @staticmethod
    def from_polar(r, theta):
        """
        Returns the Cartesian coordinates of the point.
        :param r: distance
        :param theta: angle
        :return: a point expressed with cartesian coordinates
        """
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return Point([x, y, 1])

    @staticmethod
    def intersecting(l1, l2):
        """Returns the intersection point(s) of the given pair(s) of lines."""
        return np.cross(l1, l2, axis=0).view(Point)

    def transformed(self, H):
        """Points transform contravariantly with regard to an homography."""
        return H @ self

    def to_euclidean(self, *, assumeNormalized=False):
        """
        Returns the coordinates of the points in Euclidean coordinates.

        :param assumeNormalized: if True, avoid normalizing already-normalized points.
        :return: `numpy.ndarray` containing the Euclidean coordinates [x, y].
        """
        coords = self

        if not assumeNormalized:
            coords /= coords[2]

        return coords[:2]


# Well-known points
Point.ORIGIN = Point([0, 0, 1])
Point.X = Point([1, 0, 0])
Point.Y = Point([0, 1, 0])


class Transform(Vector):
    """One or more 2D homogeneous transformation matrices. Expected shape: n⨉3⨉3."""

    # The type of the object with the highest priority among the inputs to an
    # operation determines the type of the outputs. Setting a lower-than-default
    # priority, so that Transform @ Point returns a Point as expected.
    __array_priority__ = -1.0

    identity = np.eye(3)

    @staticmethod
    def rotate(theta):
        s, c = np.sin(theta), np.cos(theta)

        return Transform([
            [ c, -s, 0],
            [+s,  c, 0],
            [ 0,  0, 1],
        ])

    @staticmethod
    def translate(tx, ty):
        return Transform([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0,  1],
        ])
