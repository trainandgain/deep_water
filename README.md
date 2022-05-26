# Detectron 2 

## Requirements

<code>conda install pytorch torchvision -c pytorch</code>
<code>pip install pyyaml==5.1</code>
*# Detectron2 has not released pre-built binaries for the latest pytorch (https://github.com/facebookresearch/detectron2/issues/4053)
# so we install from source instead. This takes a few minutes.*
<code>pip install 'git+https://github.com/facebookresearch/detectron2.git'</code>

## inference.py
<code>python3 inference.py *video_path*</code>