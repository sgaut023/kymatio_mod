"""Contains the exceptions related to the modules package

exceptions: 
    InvalidInitializationException
    InvalidArchitectureError
"""


class InvalidInitializationException(Exception):
    """Error thrown when an invalid initialization scheme is passed"""
    pass

class InvalidArchitectureError(Exception):
    """Error thrown when an invalid architecture name is passed"""
    pass