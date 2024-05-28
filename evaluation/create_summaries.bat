@echo off
:: Change this line to the path of your Anaconda/Miniconda installation
set CONDA_PATH=C:\Users\...\AppData\Local\mambaforge

:: Change this line to the name of your conda environment
set ENV_NAME=focus

:: Activate the conda environment
call %CONDA_PATH%\Scripts\activate %ENV_NAME%

:: create summary images
FOR /L %%A IN (9980,1,9999) DO (

  call python sumgen_script.py --dataset ilsvrc2012 --architecture vgg11_bn --mosaic_idx %%A --mosaic_dataset ilsvrc2012_mosaic
  call python sumgen_script.py --dataset ilsvrc2012 --architecture vgg11_bn --mosaic_idx %%A --mosaic_dataset ilsvrc2012_mosaic --info
  call python sumgen_script.py --dataset ilsvrc2012 --architecture resnet50 --mosaic_idx %%A --mosaic_dataset ilsvrc2012_mosaic
  call python sumgen_script.py --dataset ilsvrc2012 --architecture resnet50 --mosaic_idx %%A --mosaic_dataset ilsvrc2012_mosaic --info
)
FOR /L %%A IN (1,1,10) DO (
  call python sumgen_script.py --dataset carsncats --architecture vgg11_bn --mosaic_idx %%A --mosaic_dataset carsncats_mosaic
  call python sumgen_script.py --dataset carsncats --architecture vgg11_bn --mosaic_idx %%A --mosaic_dataset carsncats_mosaic --info
  call python sumgen_script.py --dataset carsncats --architecture resnet50 --mosaic_idx %%A --mosaic_dataset carsncats_mosaic
  call python sumgen_script.py --dataset carsncats --architecture resnet50 --mosaic_idx %%A --mosaic_dataset carsncats_mosaic --info

  call python sumgen_script.py --dataset mountaindogs --architecture vgg11_bn --mosaic_idx %%A --mosaic_dataset mountaindogs_mosaic
  call python sumgen_script.py --dataset mountaindogs --architecture vgg11_bn --mosaic_idx %%A --mosaic_dataset mountaindogs_mosaic --info
  call python sumgen_script.py --dataset mountaindogs --architecture resnet50 --mosaic_idx %%A --mosaic_dataset mountaindogs_mosaic
  call python sumgen_script.py --dataset mountaindogs --architecture resnet50 --mosaic_idx %%A --mosaic_dataset mountaindogs_mosaic --info
)

call mamba deactivate
