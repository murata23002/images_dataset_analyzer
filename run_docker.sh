#!/bin/bash
docker run --rm --gpus all -it \
    --name tensorflow_container2 \
    -v $(pwd):/workspace \
    -v $(pwd)/output:/workspace/output \
    -v /home/shimalab/murata_work/tensor_model_maker/object_detection/dataset/:/workspace/dataset \
     -v /home/shimalab/murata_work/data/:/workspace/murata_data \
    tensorflow-gpu-custom /bin/bash
