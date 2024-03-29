"""Exceptions related to PopNet.

"""

class PopNetError(Exception):
    """Generic class for exceptions related to PopNet."""
    pass


class FormatError(PopNetError):
    """Exceptions related to formatting errors in files handled by PopNet."""
    pass


class StopSimulation(PopNetError):
    """Exceptions raised when a stochastic simulation has to end."""
    pass


class PopNetWarning(Warning):
    """Generic class for warnings related to PopNet."""
    pass