# -*- coding: utf-8 -*-
"""统一入口：python main.py <子命令> [参数] （详见 README.md）"""
import sys

from hybrid_search.cli import main

if __name__ == "__main__":
    sys.exit(main())
