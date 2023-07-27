# ovrs

OpenVINO rust example.  
https://github.com/intel/openvino-rs/blob/main/crates/openvino/tests/classify-alexnet.rs

# Prerequisite

* Install OpenVINO runtime (2022.3.x) from [here](https://docs.openvino.ai/2022.3/openvino_docs_install_guides_installing_openvino_macos_header.html)
* Set OpenVINO environment variables
```shell
$ cd /opt/intel/openvino_2022
$ source /opt/intel/openvino_2022/setupvars.sh
$ cd <project-root>
```
* Install OpenVINO python dev tools (`python` == 3.7)
```shell
$ pip install -r requirements.txt
```
* Prepare model files
```shell
$ omz_downloader --name alexnet
$ omz_converter --name alexnet
```

# Run
```shell
$ cargo run
```
