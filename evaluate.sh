#!/bin/bash
# path to your model 
MODEL=${1}

dir=$(pwd)
export CUDA_VISIBLE_DEVICE="0"
        

# zero-shot
python ${dir}/model/main.py ${MODEL} \
        --act_sort_metric max\
        --dataset wikitext2\
        --tasks piqa,arc_challenge,boolq,hellaswag,winogrande,lambada_openai,arc_easy \
        --lm_eval_num_fewshot 0 \
        --lm_eval_limit -1\
        --quant_type NVFP4\


#5-shot mmlu
python ${dir}/model/main.py ${MODEL}\
        --act_sort_metric max\
        --dataset wikitext2\
        --tasks mmlu\
        --lm_eval_num_fewshot 5\
        --lm_eval_limit -1\
        --quant_type NVFP4\


# wikitext2 ppl
python ${dir}/model/main.py ${MODEL}\
        --act_sort_metric max\
        --dataset wikitext2\
        --lm_eval_limit -1\
        --eval_ppl\
        --quant_type NVFP4\
