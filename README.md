# 👁️ TinyImage classification

*Copied from [the report](./Final%20report.pdf).*

The work is uploaded to Github at [esbenkc/tinyimage-classification](https://github.com/esbenkc/tinyimage-classification) and the process to run it is simple. Clone the project and run `python main.py` (optionally run `pip install -r ./requirements.txt` but this will install a lot of packages). If you wish to see it in Tensorboard, run the command `tensorboard --logdir logs/fit` which will also show the runs from this report. This will reproduce the best result reported in this report, i.e. the MobileNetV2 transfer learning and fine-tuned model. This will generate the file `submission.csv` and `cnn_mobilenet_model.h5`.

The device used is a Windows 11 laptop with 32GB RAM and a CUDA GTX1650Ti with 6GB dedicated memory. To run the models on the GPU, we use CuDNN with Tensorflow GPU.