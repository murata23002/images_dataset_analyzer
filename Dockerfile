FROM tensorflow/tensorflow:2.13.0-gpu

RUN apt-get update && \
    apt-get install -y libusb-1.0-0 libusb-1.0-0-dev git libgl1-mesa-glx

# 作業ディレクトリを設定
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

# 出力ディレクトリを作成
RUN mkdir -p /workspace/dist

# TensorFlowがGPUを認識しているか確認するためのスクリプトを追加
RUN echo "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" > /workspace/check_gpu.py

# コンテナ起動時にGPUの確認を行う
CMD ["python", "/workspace/check_gpu.py"]