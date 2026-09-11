import logging

from derivatives_bt_engine.utils.logger import setup_logger


def test_package_file_handler_collects_live_and_futures_child_logs(tmp_path):
    """All package descendants must share one file sink, not sibling handlers."""
    package_logger = logging.getLogger('derivatives_bt_engine')
    original_handlers = package_logger.handlers[:]
    original_level = package_logger.level
    original_propagate = package_logger.propagate
    log_path = tmp_path / 'shared.log'

    try:
        for handler in original_handlers:
            package_logger.removeHandler(handler)

        configured = setup_logger(str(log_path))
        logging.getLogger('derivatives_bt_engine.live.tsmom_rebalance').debug('live decision')
        logging.getLogger('derivatives_bt_engine.strats.naked_futures').info('futures decision')
        for handler in configured.handlers:
            handler.flush()

        assert len(configured.handlers) == 1
        contents = log_path.read_text()
        assert 'live decision' in contents
        assert 'futures decision' in contents
    finally:
        for handler in package_logger.handlers[:]:
            package_logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            package_logger.addHandler(handler)
        package_logger.setLevel(original_level)
        package_logger.propagate = original_propagate
