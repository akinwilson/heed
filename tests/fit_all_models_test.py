"""
Tests whether all models in model zoo can be fitted; trained, validated and tested, and there model weights are saved correctly.
"""
import sys
import os
from pathlib import Path

path = str(Path(__file__).parent.parent)
print(path)


def test_resnet():
    os.environ["ENV_FILE_PATH"] = "./env_vars/.resnet.test.env"
    sys.path.insert(1, path)
    from fit import main

    main()


def test_leenet():
    os.environ["ENV_FILE_PATH"] = "./env_vars/.leenet.test.env"
    sys.path.insert(1, path)
    from fit import main

    main()


if __name__ == "__main__":
    test_resnet()
    test_leenet()
