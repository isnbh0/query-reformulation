import loguru as loguru
import sys
import logging

LOGGING_FORMAT = (
    "<green>{time:YYYY-MM-DDTHH:mm:ss.SSSZ}</green> | "
    "<level>{level: >8}</level> | "
    "<cyan>{file}</cyan>:<cyan>{line}</cyan> | "
    "<cyan>{module}</cyan>| "
    "<cyan>{function}</cyan>|> "
    "<level>{message}</level>"
)

LOGGER = loguru.logger
LOGGER.remove()
LOGGER.add(sink=sys.stdout, level='TRACE', format=LOGGING_FORMAT, enqueue=True, diagnose=True)

# Redirect standard logging to loguru
# class InterceptHandler(logging.Handler):
#     def emit(self, record):
#         # Get corresponding Loguru level if it exists
#         try:
#             level = LOGGER.level(record.levelname).name
#         except ValueError:
#             level = record.levelno

#         # Find caller from where originated the log message
#         frame, depth = logging.currentframe(), 2
#         while frame.f_code.co_filename == logging.__file__:
#             frame = frame.f_back
#             depth += 1

#         LOGGER.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# # Set up logging for the beir module
# beir_logger = logging.getLogger('beir')
# beir_logger.handlers = [InterceptHandler()]
# beir_logger.setLevel(logging.DEBUG)  # Adjust the level as needed

# # If beir uses its own logger, you might need to set it up similarly
# # Example: beir_logger = logging.getLogger('beir')
# # beir_logger.handlers = [InterceptHandler()]