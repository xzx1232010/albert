# albert_tfserving_docker

#### fine_tune（微调模型，用于下游任务）
```
python3 run_classifier.py    --task_name=lcqmc_pair \  
                             --do_train=true \  
                             --do_eval=true \  
                             --data_dir=./lcqmc \   
                             --vocab_file=./albert_config/vocab.txt  \
                             --bert_config_file=./albert_tiny/albert_config_tiny.json  \
                             --max_seq_length=64  \
                             --train_batch_size=64  \ 
                             --learning_rate=1e-4  \
                             --num_train_epochs=1 \
                             --output_dir=./albert_lcqmc_checkpoints \
                             --init_checkpoint=./albert_tiny/albert_model.ckpt  
                             --do_export=true \
                             --export_dir=./export
```
#### 查看模型结构的节点名称
```
可以通过结合输出目录下的graph.pbtxt文件和模型代码来查看。
```
#### 冻结模型，转为pb格式
```
freeze_graph  --input_checkpoint=./albert_lcqmc_checkpoints/model.ckpt-100 \
              --output_graph=./albert_lcqmc_checkpoints/albert_tiny_zh.pb \
              --output_node_names=loss/probabilities \
              --input_binary=True \
              --input_meta_graph=./albert_lcqmc_checkpoints/model.ckpt-100.meta
```
#### 单行预测
```
python3 inference.py
```
#### 查看saveModel格式的数据结构
```
saved_model_cli show --all --dir ./export/1576720765
```
#### 以batch方式运行docker_tfserving_gpu
```
docker run --runtime=nvidia -p 9000:8500 --gpus '"device=1"' \
  --mount type=bind,source=./export,target=/models/test \
  -e MODEL_NAME=test -t tensorflow/serving:1.14.0-gpu --enable_batching=true \
  --per_process_gpu_memory_fraction=0.5 --batch_timeout_micros=1000 \
  --max_enqueued_batches=1000 --tensorflow_session_parallelism=2 \
  --num_batch_threads=8 --max_batch_size=9 &
  
参数说明：
--gpus：指定哪块gpu设备
--enable_batching：开启batch模式
--per_process_gpu_memory_fraction：限制显存使用
--batch_timeout_micros：形成batch的等待时间，如果在客户端没有合并成bacth，就需要等待多个请求合并成batch，建议设置小一点，几毫秒即可
--max_enqueued_batches：队列的最大batches的数目
--tensorflow_session_parallelism：并行的seesion会话数
--num_batch_threads：并行度，并发处理batches的最大数目
--max_batch_size：batch的大小，影响吞吐/延迟
```
#### grpc客户端代码预测
```
python3 grpc_batching.py
```