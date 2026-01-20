"""
Compatibility patch for transformers 4.45.2
Provides missing use_kernel_forward_from_hub decorator
"""

def use_kernel_forward_from_hub(kernel_name):
    """
    Dummy decorator for compatibility with older transformers versions.
    In newer versions, this decorator enables kernel optimization from hub.
    For testing purposes, we just return the class unchanged.
    """
    def decorator(cls):
        return cls
    return decorator
