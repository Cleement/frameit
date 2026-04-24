# Copyright 2026 Clément Soufflet, Météo-France
# Licensed under the Apache License, Version 2.0
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0

import logging

from ._version import __version__

logging.getLogger("frameit").addHandler(logging.NullHandler())
