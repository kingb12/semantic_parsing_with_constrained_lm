export PRETRAINED_MODEL_DIR=facebook/bart-base
export TRAINED_MODEL_DIR=trained_models/

for domain in "basketball" "blocks" "calendar" "housing" "publications" "recipes" "restaurants" "socialnetwork"; do
# for domain in "basketball"; do
    python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
          --exp-names overnight_${domain}_utterance \
          --lr 1e-6 \
          --num-steps 12000 \
          --steps-per-save 1000 \
          --model-type BartV3 \
          --steps-per-decay 8 \
          --batch-size 32

done