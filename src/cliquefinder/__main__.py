"""Entry point for running cliquefinder as a module: python -m cliquefinder"""

from cliquefinder.cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
