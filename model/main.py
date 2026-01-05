import torch
from collections import defaultdict

from model_utils import reorder_model_llama, reorder_model_qwen
from parallel_utils import map_layers_to_multi_gpus
from datautils import get_loaders
from eval import *

from lm_eval import tasks as lm_tasks
from lm_eval import evaluator as lm_evaluator
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM

import time

from visualize import *


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    # model.seqlen = 2048
    return model

def get_qwen(model):
    import torch
    def skip(*args, **kwargs):
        pass
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
   
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, 
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--act_sort_metric', type=str, default='max', choices=['mean', 'frobenius', 'hessian', 'max'],
        help='The metric used to sort the activations.'
    )
   
    parser.add_argument(
        '--kv_cache', action='store_true',
        help='Whether to quant KV_Cache'
    )

    parser.add_argument(
        '--tasks', type=str, default=None,
    )
    parser.add_argument(
        "--eval_ppl", action="store_true",
        help='Whether to evaluate perplexity.'
    )

    parser.add_argument(
        "--lm_eval_num_fewshot", type=int, default=0, 
        help="Number of shots in lm evaluation. Default is 0 for zero-shot."
    )
    parser.add_argument(
        "--lm_eval_limit", type=int, default=-1, 
        help="Limit the number of examples in lm evaluation"
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "pile", "humaneval"], 
        help="The calibration dataset to use."
    )
    parser.add_argument(
        "--quant_type", type=str, default="NVFP4", choices=["NVFP4", "MXFP4", "INT4"], 
        help="data type for W and A quantization."
    )
  
    
    args = parser.parse_args()

    model_name = args.model.split('/')[-2] if len(args.model.split('/')[-1]) == 0 else args.model.split('/')[-1]
    assert model_name != None, "Please check the model path."

    if "llama" in args.model.lower():
        model = get_llama(args.model)
        reorder_model_func = reorder_model_llama
       
    elif "qwen" in args.model.lower():
        model = get_qwen(args.model)
        reorder_model_func = reorder_model_qwen
       
    model.eval()

    import os

    dataset_name = args.dataset.lower()
    index_filename = f'./saved/{model_name.lower()}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt'
    select_num_filename = f'./saved/{model_name.lower()}_select_num_{dataset_name}_{args.act_sort_metric}.pt'
    act_scales_filename = f'./saved/{model_name.lower()}_act_scales_{dataset_name}_{args.act_sort_metric}.pt'
 
    
    assert os.path.isfile(index_filename), "reorder index file not found."

    print("Loading cached reording index from disk...")
    reorder_index = torch.load(index_filename, weights_only=False)
    select_nums = torch.load(select_num_filename, weights_only=False)
    act_scales = torch.load(act_scales_filename, weights_only=False)

    
    torch.cuda.reset_max_memory_allocated()
    print("Reordering model...")
    start_time=time.time()
    model = reorder_model_func(
        model, device=DEV, kv_cache=args.kv_cache, reorder_index=reorder_index, select_nums=select_nums, quant_type=args.quant_type
    )
    end_time=time.time()
    peak_memory = torch.cuda.max_memory_allocated()


    print(model)
    print(f"Quantized Model Size: {peak_memory/(1024*1024*1024):.2f} GB")
    print(f"Quantized Type is: {args.quant_type} ")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    bsz = "auto"
    if args.tasks is not None:
        if 'mmlu' in args.tasks :
            bsz = 4
 
    from transformers import AutoTokenizer
    lm = HFLM(model, batch_size=bsz)

    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    # map_layers_to_multi_gpus(lm.model.model.layers)
    # input_device = lm.model.model.layers[0].device
    # output_device = lm.model.model.layers[-1].device
    # assert input_device == output_device
    # lm._device = input_device
    # lm.model.model.embed_tokens.to(input_device)
    # lm.model.model.norm.to(output_device)
    # lm.model.lm_head.to(output_device)
    lm._device = DEV
    lm._model = lm._model.to(lm._device)

        
    if args.eval_ppl:
        datasets = ['wikitext2']

        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=2048
            )
            print(f"Evaluating {dataset} ...")
            ppl = eval_ppl(lm.model, testloader, 'cuda')

            print(f"Result,{dataset},{ppl:.3f}")

    
            
    if args.tasks is not None:
        task_manager = TaskManager()
        task_names = args.tasks.split(',')

        results = lm_evaluator.simple_evaluate(
            lm,
            tasks=task_names,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=None if args.lm_eval_limit == -1 else args.lm_eval_limit,
            batch_size=bsz
        )

        table_results = make_table(results)
        print(table_results)
        import logging
        from datetime import datetime

        if not os.path.exists("./results/"):
            os.makedirs("./results/")
        log_filename = f"./results/log_{model_name.lower()}_{args.tasks}_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
                            filename=log_filename,
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                        )
        logging.info(f"Results for {model_name.lower()} on {args.tasks}:\n{table_results}")
  