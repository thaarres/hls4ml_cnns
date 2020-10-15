python3 -m pip freeze --user --exclude-editable > pip_requisites.txt --user
python3 -m pip install git+https://github.com/google/qkeras.git@master --user
python3 -m pip install git+https://github.com/vloncar/hls4ml.git@cnn_stream --user
