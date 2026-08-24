import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fwi_fars


if __name__ == "__main__":
    fwi_fars.main()
