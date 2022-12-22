
# Demo DacovaSegmentation

## 1. Install libs + download weight
```
pip install -r requirements.txt
```

## 2. Run infer openvino-python
```
sh scripts/infer.sh
```
## 3. Run infer openvino-cpp
```
docker build cpp -t openvino_segmentation
docker run -it --rm -v $(pwd):/segment-openvino openvino_segmentation

cd cpp 
mkdir build 
cd build
cmake ../ -O ./
make 
./main
```

