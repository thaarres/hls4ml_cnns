# hls4ml_cnns
CNN firmware using hls4ml. To run on hlssynth-04.

Log in from within lxplus firewall
```python
ssh -XY myuser@hlssynt-04
```
copy models from AFS to Docker repository (you will have no access to. your AFS from docker)
```python
cd /data/myuser/hls4ml_docker/
cp /path/to/mymodel.h5 models/
```
Get hls4ml's CNN branch
```python
pip install git+https://github.com/vloncar/hls4ml.git@cnn_stream
cd hls4ml
pip3 install . --user
```
Now get Vivado through docker
```python
docker exec -it --user `id -u`:`id -g` -e DISPLAY=$DISPLAY hls4ml-test bash
```
and run
```python
python3 benchmarkModels.py mymodel.h5
```
