class try_import:

    def __init__(self):
        self.success = False

    def __bool__(self):
        return self.success

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is ImportError or exc_type is ModuleNotFoundError:
            # Catch import errors
            return True
        elif exc_type is not None:
            # Propogate other exceptions
            return False

        # No import errors, so return
        self.success = True
        return True

    def __str__(self):
        return str(self.success)
