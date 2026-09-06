import time


class Timer:
    """
    Count the time of defined process and print it.
    """
    def __init__(self, 
                 label:str,
                 shift_number:int = 0):
        """
        Initiate the timer

        Parameters
        ----------
        label : str
            Describtion of the process that timer will measure
        shift_number : int, optional
            Number of shifts in text with which 
            the communicate will be printed in console, 
            by default 0
        """
        self.label = label
        self.shift_number = shift_number

    def __enter__(self):
        """
        Start the time measurement
        """
        self.t = time.perf_counter()
    
    def __exit__(self, *_):
        """
        Finish the time measurement and print the results
        """
        print("  "*self.shift_number+ f"  · {self.label}: {time.perf_counter() - self.t:.4f}s")