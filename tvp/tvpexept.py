class TVPError(Exception):
    pass

class NotMonotonicIncreasingIndex(TVPError):
    """Raised when a datetime index is not monotonic increasing"""
    pass

class NotFixedFrequencyIndex(TVPError):
    """Raised when a datetime index has no fixed frequency"""
    pass
