import sys


def error_message_detail(error, error_detail: sys) -> str:
    """Build a detailed message: which file, which line, and the error text."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_massage = f"Error in script [{file_name}] at line [{line_number}]: {error}"
    return error_massage


class CustomException(Exception):
    def __init__(self, error_message_from_exception: str, error_detail_from_run_time: sys):
        super().__init__(error_message_from_exception)
        # store the detailed, location-aware message
        self.error_message = error_message_detail(error_message_from_exception, error_detail_from_run_time)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    pass
