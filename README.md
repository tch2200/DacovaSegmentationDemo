
# Demo DacovaSegmentation

## 1. Install libs
```
pip install -r requirements.txt
```

## 2. Run infer openvino python
```
python infer_openvino.py
```
## 3. Run infer onnx python
```
python infer_onnx.py
```
## 4. Run infer openvino-cpp
```
docker build openvino_cpp -t openvino_segmentation
docker run -it --rm -v $(pwd):/segment-openvino openvino_segmentation

cd segment-openvino/openvino_cpp 
mkdir build 
cd build
cmake ../ -O ./
make 
./main
```
## 5. Run infer onnx-cpp
```
docker build onnx_cpp -t onnx_segmentation
docker run -it --rm -v $(pwd):/segment-onnx onnx_segmentation

cd segment-onnx/onnx_cpp 
mkdir build 
cd build
cmake ../ -O ./
make 
./main
```

