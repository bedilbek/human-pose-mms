FROM awsdeeplearningteam/multi-model-server

USER root

RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1-mesa-glx \
    libgtk2.0-dev

RUN pip install mxnet-mkl==1.6.0 certifi==2020.06.20 gluoncv==0.8.0 opencv-python==4.4.0.42

USER model-server

# I hate when after passage of time people get stuck in issues of package incompatability,
# so in order to prevent that kind of blaming towards my side let's write what versions of packages and images I used:
# Docker images:
#  - awsdeeplearningteam/multi-model-server:latest - DIGEST:sha256:b441cc6ec7ea882fbef85162b07c9f2ced6b935eaf70e8d085e0f47218a0263e
# Most important Python packages:
#  - certifi==2020.6.20
#  - numpy==1.19.1
#  - matplotlib==3.3.1
