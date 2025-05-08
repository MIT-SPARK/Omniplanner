import logging


# adapted from https://gist.github.com/ablakey/4f57dca4ea75ed29c49ff00edf622b38
class RosForwarder(logging.Handler):
    """Class to forward logging to ros handler."""

    def __init__(self, node, **kwargs):
        """Construct a logging Handler that forwards log messages to ROS."""
        super(RosForwarder, self).__init__(**kwargs)
        self._level_map = {
            logging.DEBUG: node.get_logger().debug,
            logging.INFO: node.get_logger().info,
            logging.WARNING: node.get_logger().warning,
            logging.ERROR: node.get_logger().error,
            logging.CRITICAL: node.get_logger().fatal,
        }

    def emit(self, record):
        """Send message to ROS."""
        level = (
            record.levelno if record.levelno in self._level_map else logging.CRITICAL
        )
        self._level_map[level](f"{record.name}: {record.msg}")
        self._level_map[logging.CRITICAL](f"{record.name}: {record.msg}")


def setup_ros_log_forwarding(node, py_logger, level=logging.INFO):
    """Forward logging to ROS."""
    py_logger.addHandler(RosForwarder(node))
    py_logger.setLevel(level)
