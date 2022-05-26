# Detectron 2 
## Requirements

<code> 
conda install pytorch torchvision -c pytorch
pip install pyyaml==5.1
# Detectron2 has not released pre-built binaries for the latest pytorch (https://github.com/facebookresearch/detectron2/issues/4053)
# so we install from source instead. This takes a few minutes.
pip install 'git+https://github.com/facebookresearch/detectron2.git'
</code>