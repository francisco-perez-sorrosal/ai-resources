import logging

import project_logging
from deep_lib.model_structure import LLayerModel

project_logging.setup_logging(show_config=True)

logger = logging.getLogger(__name__)
logger.info('Completed configuring logger for module: %s', __name__)