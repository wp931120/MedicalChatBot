from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Mapping
from transformers import HfArgumentParser
from loguru import logger

MODEL_CLASSES = ["baichuan", "qwen"]
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES)}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

if __name__ == '__main__':
   print(ModelArguments.model_name_or_path)
   print(ModelArguments.model_type)
