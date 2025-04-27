# setup.py
from setuptools import setup, find_packages

setup(
    name="checkers_gym",
    version="0.1",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy"],
    entry_points={
        "gymnasium.envs": [
            "Checkers-v0 = gym_env:CheckersGymEnv",
        ]
    }
)
