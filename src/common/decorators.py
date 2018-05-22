def time_this(function: callable) -> callable:
    """ Print the execution time of the wrapped function. """
    def wrapper(*args, **kwargs):
        from time import time
        time_begin = time()
        result = function(*args, **kwargs)
        time_end = time()
        time_total = time_end - time_begin
        second_or_seconds = "second" if (time_total < 1) else "seconds"
        print("Execution time for \"{}\": {} {}".format(
            function.__name__, time_total, second_or_seconds))
        return result
    return wrapper


if __name__ == "__main__":
    pass
