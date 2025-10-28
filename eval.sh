MODEL_NAME=$1
DOMAIN=$2
NGPUS=${3:-1}  # Default to 1 GPU
RESULTS_DIR=${4:-"./response/$DOMAIN"}
EVAL_DIR=${5:-"./eval_results/$DOMAIN"}

if [ $DOMAIN == "math" ]
then
    TEST_DATASETS="aime2024 aime2025 amc23 math500 minerva olympiad_math" 
elif [ $DOMAIN == "science" ]
then
    TEST_DATASETS="gpqa_diamond rbench arc_challenge scibench"
elif [ $DOMAIN == "logic" ]
then
    TEST_DATASETS="ProntoQA ProofWriter FOLIO LogicalDeduction AR-LSAT"
elif [ $DOMAIN == "coding" ]
then
    TEST_DATASETS="MBPP"
else
    echo "Unsupported domain: $DOMAIN"
    exit 1
fi

for DATASET in $TEST_DATASETS
do
    echo "Generating on $DATASET dataset..."
    python3 eval/inference.py \
        --model_path $MODEL_NAME \
        --dataset $DATASET \
        --tp $NGPUS \
        --output_dir ${RESULTS_DIR}
done

echo "Generation completed. Starting evaluation..."

python3 eval/eval.py \
    --response_path ${RESULTS_DIR} \
    --eval_path ${EVAL_DIR}

echo "Evaluation completed. Results are saved in ${EVAL_DIR}."

echo "calculating overall score..."
python3 eval/calc_passk.py \
    --eval_path ${EVAL_DIR} \
    --k "1,32"