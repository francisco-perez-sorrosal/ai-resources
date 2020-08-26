import logging.config
import os
from pprint import pprint

import yaml

### References:
### http://www.patricksoftwareblog.com/python-logging-tutorial/
### https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
def setup_logging(
    default_path='../logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG',
    show_config=False
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            if show_config:
                print("Logging configuration from file %s:" % f.name)
                pprint(config)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)