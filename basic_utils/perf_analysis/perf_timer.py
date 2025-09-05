import logging
import time
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@contextmanager
def perf_timer(task_name: str):
    """上下文管理器：打印任务耗时"""
    start = time.perf_counter()
    logger.info("Start %s", task_name)
    yield
    elapsed = time.perf_counter() - start
    logger.info("Finished %s in %.2f s", task_name, elapsed)
