import time

def current_timestamp_ms():
    """Get the current timestamp in milliseconds.

    Returns:
        int: The current timestamp in milliseconds.
    """
    return round(time.time() * 1000)
