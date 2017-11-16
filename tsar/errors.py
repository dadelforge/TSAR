# User defined exceptions
class TVPError(Exception):
    """Base class for Time Value Paired model errors"""
    pass


class NotMonotonicIncreasingError(TVPError):
    """Raised when a datetime index is not monotonic increasing"""
    pass


class NotFixedFrequencyError(TVPError):
    """Raised when a datetime index has no fixed frequency"""
    pass
