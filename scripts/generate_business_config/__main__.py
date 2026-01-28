#!/usr/bin/env python3
# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Entry point for running the business configuration generator as a module.

Usage:
    python -m scripts.generate_business_config --help
"""

from .cli import main

if __name__ == "__main__":
    main()
