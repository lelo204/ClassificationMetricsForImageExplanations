@echo off
:: Change this line to the path of your Anaconda/Miniconda installation
set CONDA_PATH=C:\Users\...\AppData\Local\mambaforge

:: Change this line to the name of your conda environment
set ENV_NAME=focus

:: Activate the conda environment
call %CONDA_PATH%\Scripts\activate %ENV_NAME%

:: compute alpha for the xai-methods with negative feature importance
call python compute_viz_alphas.py --dataset mountaindogs --architecture resnet50 --xai_methods bcos shap intgrad lrp --negative_FI True
call python compute_viz_alphas.py --dataset mountaindogs --architecture vgg11_bn --xai_methods bcos shap intgrad lrp --negative_FI True
call python compute_viz_alphas.py --dataset carsncats --architecture resnet50 --xai_methods bcos shap intgrad lrp --negative_FI True
call python compute_viz_alphas.py --dataset carsncats --architecture vgg11_bn --xai_methods bcos shap intgrad lrp --negative_FI True
call python compute_viz_alphas.py --dataset ilsvrc2012 --architecture resnet50 --xai_methods bcos shap intgrad lrp --negative_FI True
call python compute_viz_alphas.py --dataset ilsvrc2012 --architecture vgg11_bn --xai_methods bcos shap intgrad lrp --negative_FI True

call python compute_viz_alphas.py --dataset mountaindogs --architecture resnet50 --xai_methods bcos gradcam gradcam++ smoothgrad intgrad lime lrp shap
call python compute_viz_alphas.py --dataset mountaindogs --architecture vgg11_bn --xai_methods bcos gradcam gradcam++ smoothgrad intgrad lime lrp shap
call python compute_viz_alphas.py --dataset carsncats --architecture resnet50 --xai_methods bcos gradcam gradcam++ smoothgrad intgrad lime lrp shap
call python compute_viz_alphas.py --dataset carsncats --architecture vgg11_bn --xai_methods bcos gradcam gradcam++ smoothgrad intgrad lime lrp shap
call python compute_viz_alphas.py --dataset ilsvrc2012 --architecture resnet50 --xai_methods bcos gradcam gradcam++ smoothgrad intgrad lime lrp shap
call python compute_viz_alphas.py --dataset ilsvrc2012 --architecture vgg11_bn --xai_methods bcos gradcam gradcam++ smoothgrad intgrad lime lrp shap

call mamba deactivate