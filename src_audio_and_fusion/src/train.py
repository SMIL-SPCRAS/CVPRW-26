import sys
from chimera_ml.cli import app as chimera_main


def main():
    argv = sys.argv[1:]
    chimera_main(argv)


if __name__ == "__main__":
    main()