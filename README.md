# hls4ml_cnns
CNN firmware using hls4ml. To run on hlssynth-04.

Log in from within lxplus firewall
```python
ssh -XY myuser@hlssynt-04
```
copy models from AFS to Docker repository (you will have no access to. your AFS from docker)
```python
cd /data/myuser/hls4ml_docker/
git clone git@github.com:thaarres/hls4ml_cnns.git
cd hls4ml_cnns
pip install -r pip_prerequisites.txt --user
cp /path/to/mymodel.h5 models/
```
Get hls4ml's CNN branch and qkeras
```python
pip install git+https://github.com/google/qkeras.git@master
git clone -b cnn_stream git@github.com:vloncar/hls4ml.git
cd hls4ml
pip install . --user
```
Now get Vivado through docker
```python
docker exec -it --user `id -u`:`id -g` -e DISPLAY=$DISPLAY hls4ml-test bash
cd /data/myuser/hls4ml_docker/
```
and run
```python
python3 benchmarkModels.py full.h5
python3 benchmarkModels_qkeras.py quant
```
To plot accuracy, resources and latency
```python
python3 hls4ml_scans.py
```