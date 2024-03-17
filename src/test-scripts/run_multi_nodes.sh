#!/bin/bash
### GPUS
GPUS=8
RUN_HOSTS=(g0285 g0286)
#RUN_HOSTS=(g0285)

### 脚本名称
RANK_SCRIPT="rank-7b.sh"

### Job Path
JOB_PATH=`pwd`

### Job ID
JOB_ID=`date +"%y%m%d%H%M%S"`
mkdir ${JOB_ID}

### hosfile
HOSTFILE="${JOB_ID}/hostfile"

### 获取节点主机名
for i in "${RUN_HOSTS[@]}";
do
  let k=k+1
  host[$k]=$i
  rank[$k]=$(($k-1))
  echo "${host[$k]} slots=$GPUS" >> $HOSTFILE
done

### 设置主节点,将第一个节点主机名做为 master 地址.
MASTER_ADDR=${host[1]}

### Nodes
NODES="${#host[@]}"

### 清理可能存在的残留进程.
/usr/bin/pkill -9 python
for((i=2;i<=${NODES};i++));  
do
   node_host=${host[$i]}
   pdsh -w ssh:"${node_host}" "/usr/bin/pkill -9 python"
done

### nodes gpus rank master_addr hostfile job_id
bash ${RANK_SCRIPT} ${NODES} ${GPUS} 0 ${MASTER_ADDR} ${HOSTFILE} ${JOB_ID} &
for((i=2;i<=${NODES};i++));
do
   node_host=${host[$i]}
   node_rank=${rank[$i]}
   echo "nodes:${NODES}, host:${node_host}, node_rank:${node_rank}, master_addr:${MASTER_ADDR}"
   pdsh -w ssh:"${node_host}" "cd ${JOB_PATH} ; /bin/bash ${RANK_SCRIPT} ${NODES} ${GPUS} $node_rank ${MASTER_ADDR} ${HOSTFILE} ${JOB_ID}" &
done
wait
