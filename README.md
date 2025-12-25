# Introduction

This is the code repository for my ESVAD thesis paper, which should be able to reproduce exactly the result wrote in the paper. This repository does not cover the running of comparison projects on my dataset. Those can be found at my forked [GNN-ReGVD](https://github.com/icewanee/GNN-ReGVD-thesis-dataset), [AIBugHunter](https://github.com/icewanee/AIBugHunter-thesis-dataset) and [Devign](https://github.com/icewanee/devign-thesis-dataset). Where I dockerized the projects for maximum reproducibility as the original repository is not runnable anymore.

# Environment setup

Your machine needs Nvidia GPU with at least 24GB of VRAM, we used RTX4090 with Cuda 12.4. It is highly recommended to use Docker with my [Dockerhub Image](https://hub.docker.com/r/icewanee/esvad). We ran this on Runpod.io's infrastructure with the Docker image. For Runpod-specific setup please follow [this guide](README.runpod.md). **You'll need ~400GB of disk space for Juliet dataset experiment and ~200GB to run the SARD dataset experiment.**

If you decided not to use the docker image, you'll need to try to follow the commands in Dockerfile from a fresh ubuntu installation. Note that since the dependencies may change all the time, it is not guaranteed to work over time. With this route, you may need a lot of modifications to make things work(just like those other many research papers).

If you use Runpod.io, great! It should work right away. If you choose to run docker locally, do

```
sudo docker run --gpus=all -d icewanee/esvad:latest
```

To run the container in the background, you'll get container id out. Then,

```
sudo docker exec -it <container_id> /bin/bash
```

You should get a root shell. This is where you'll start if you start on Runpod.

To do a final check, run `nvidia-smi` and make sure your Nvidia GPU shows up. If it does, you are good to go!

# Running the experiment

## Step 1. Running things in screen

To prevent accidentally closing the bash shell or interupted remote shell connection, all the command should be run in `screen`.

1. First, spin up screen with `screen -U -s /bin/bash`
2. Once inside, start logging the screen by pressing `Ctrl+A` then `Shift+H`, it will say that it is logging the screen to `screen.log.0` or something, and will continuously do so.
3. To exit from the screen, do `Ctrl+A` then `D`
4. To go back into the screen, input `screen -r`

All the next steps shall be run in the screen to prevent losses

## Step 2. Running Juliet dataset experiment

Make sure you are in a `screen`, and quit the screen(`Ctrl+A` then `D`) if you are not using it.

1. `cd` into `/app/juliet_dataset`
2. (Optional)If you want to simulate since the code compilation steps and CPG generation, run `source process_juliet_dataset.sh`, note that this can take many hours, potentially overnight as CPG generations are slow.
3. To speed things up, we provided preprocessed CPGs, you can `source download_preprocessed_cpg.sh` to automatically download and extract the cpgs
4. `cd` back to `/app` and do `source run_juliet.sh`
5. It will take like 24-72 hours to run
6. (Optional) Before running step 9, to get the best result, you might need to adjust the threshold manually from the step 8 comparison table.
7. Results will be seen in `/app/juliet_dataset/Results` and the screenlog

## Step 3. Running SARD dataset experiment

Make sure you are in a `screen`, and quit the screen(`Ctrl+A` then `D`) if you are not using it.

1. `cd` into `/app/realsard_dataset`
2. Since SARD dataset cannot easily be compiled in a single Docker environment, we only provide preprocessed cpgs. If you want to compile the SARD dataset you'll need to find the correct `clang-15` flags and environment yourself, you can dig that through the juliet dataset processing code. To download the CPGs, run `source download_prepocessed_cpg.sh`
3. `cd` back to `/app` and do `source run_realsard.sh`
4. It will take like 24-72 hours to run
5. (Optional) Before running step 9, to get the best result, you might need to adjust the threshold manually from the step 8 comparison table.
6. Results will be seen in `/app/realsard_dataset/Results` and the screenlog
