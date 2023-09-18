import os

def clear_console():
    """
    Clear the console screen.

    This function detects the operating system and clears the console screen
    using the appropriate command. On Windows (os.name == 'nt'), it uses 'cls',
    and on Unix-like systems (including Linux and macOS), it uses 'clear'.

    Note:
        This function only works in a terminal or console environment.

    Example:
        To clear the console screen, simply call `clear_console()`.

    """
    os.system('cls' if os.name == 'nt' else 'clear')
