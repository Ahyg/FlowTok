echo "exp name: ${EXP_NAME}"

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

export WORKER_NUM=4
export TOTAL_PROC=32

echo "total workers: ${WORKER_NUM}"
echo "total gpus: ${TOTAL_PROC}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

echo "Start training"

CONFIG_PATH=$1

accelerate launch --num_machines=$WORKER_NUM --num_processes=$TOTAL_PROC --machine_rank=$WORKER_ID \
    --main_process_ip=$METIS_WORKER_0_HOST --main_process_port=$port --same_network --mixed_precision bf16 \
    train_t2i.py --config="$CONFIG_PATH"