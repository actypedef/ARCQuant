#!/bin/bash
# path to your model 
MODEL=${1}

dir=$(pwd)
export CUDA_VISIBLE_DEVICE="0"

# zero-shot
python ${dir}/model/main.py ${MODEL} \
        --act_sort_metric inf\
        --dataset wikitext2\
        --kv_cache\
        --tasks piqa,arc_challenge,arc_easy,boolq,hellaswag,winogrande,lambada_openai \
        --lm_eval_num_fewshot 0 \
        --lm_eval_limit -1\


# # wikitext2 ppl
python ${dir}/model/main.py ${MODEL}\
        --act_sort_metric inf\
        --dataset wikitext2\
        --kv_cache\
        --lm_eval_limit -1\
        --eval_ppl\


#5-shot mmlu
python ${dir}/model/main.py ${MODEL}\
        --act_sort_metric inf\
        --dataset wikitext2\
        --kv_cache\
        --tasks mmlu\
        --lm_eval_num_fewshot 5\
        --lm_eval_limit -1\

# # zero-shot
# python ${dir}/model/main.py ${MODEL} \
#         --act_sort_metric inf\
#         --dataset c4\
#         --tasks piqa,arc_challenge,arc_easy,boolq,hellaswag,winogrande,lambada_openai \
#         --lm_eval_num_fewshot 0 \
#         --lm_eval_limit -1\


# # # wikitext2 ppl
# python ${dir}/model/main.py ${MODEL}\
#         --act_sort_metric inf\
#         --dataset c4\
#         --lm_eval_limit -1\
#         --eval_ppl\


# #5-shot mmlu
# python ${dir}/model/main.py ${MODEL}\
#         --act_sort_metric inf\
#         --dataset c4\
#         --tasks mmlu\
#         --lm_eval_num_fewshot 5\
#         --lm_eval_limit -1\