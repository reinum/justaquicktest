# Osu! AI Replay Maker
# Main package initialization

__version__ = "0.1.0"
__author__ = "AI Assistant"
__description__ = "AI system for generating human-like osu! replays with controllable accuracy"

from . import data
from . import models
from . import training
from . import generation
from . import evaluation
from . import utils

__all__ = ['data', 'models', 'training', 'generation', 'evaluation', 'utils']