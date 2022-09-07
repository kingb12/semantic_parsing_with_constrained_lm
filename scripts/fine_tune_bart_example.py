import os
import wandb
from semantic_parsing_with_constrained_lm.finetune.lm_finetune import main, ModelForFineTuning

os.environ['PRETRAINED_MODEL_DIR'] = "facebook/bart-base"
os.environ['TRAINED_MODEL_DIR'] = "trained_models"

for domain in ["basketball", "blocks" "calendar", "housing", "publications", "recipes", "restaurants", "socialnetwork"]:
    """
        python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
              --exp-names overnight_${domain}_utterance \
              --lr 1e-6 \
              --num-steps 12000 \
              --steps-per-save 1000 \
              --model-type BartV3 \
              --steps-per-decay 8 \
              --batch-size 32
    """
    wandb.init(project="semantic_parsing_with_constrained_lm", entity="kingb12")
    main(lr=1e-6, exp_names=f"overnight_{domain}_utterance", num_steps=12000, warmup_steps=1000, steps_per_decay=2,
         steps_per_eval=50, steps_per_display=50, steps_per_save=1000, batch_size=32, use_context=False,
         model_type=ModelForFineTuning.BARTv3)