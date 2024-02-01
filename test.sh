set -x
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PATH=$SCRIPT_DIR:$PATH
bs=128
model=70b
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export MAX_INPUT_SEQ_LEN=1024
export MAX_TOTAL_TOKENS=2048
export BATCH_BUCKET_SIZE=$bs
export PREFILL_BATCH_BUCKET_SIZE=1
export LIMIT_HPU_GRAPH=false
max_batch_total_tokens=$(($BATCH_BUCKET_SIZE*$MAX_TOTAL_TOKENS))
export TGI_PROFILER_ENABLED=true
export SKIP_TOKENIZER_IN_TGI=true
export LOG_LEVEL=debug

export QUANT_CONFIG=/software/users/gmorys/mlperf_inference/code/llama/hqt/llama2-70b-8x/config_meas_maxabs_quant_MAXABS_HW.json
#export LOG_LEVEL_PT_SYNHELPER=0
#export GRAPH_VISUALIZATION=1
#export ENABLE_GVD=1
#export ENABLE_EXPERIMENTAL_FLAGS=true
#export ENABLE_PARTIAL_GVD=true

python /software/users/gmorys/mlperf_inference/code/llama/main.py --scenario Offline --total-sample-count 140 --model-type llama-v2-70b --accuracy --max-num-threads 256 && echo "}" >> server_events.json && sed -i 's/]\[/,/g' server_events.json
#python /nfs/trees/npu-stack/mlperf_inference/code/llama/main.py --scenario Offline --total-sample-count 128 --model-type llama-v2-70b --accuracy --max-num-threads 256
#python /nfs/trees/npu-stack/mlperf_inference/code/llama/main.py --scenario Offline --total-sample-count 128 --model-type llama-v2-7b --accuracy --max-num-threads 20
#python /nfs/trees/npu-stack/mlperf_inference/code/llama/main.py --scenario Offline --total-sample-count 1 --model-type llama-v2-7b --accuracy --max-num-threads 20
