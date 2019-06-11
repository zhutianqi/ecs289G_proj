# ECS289G Project

## Run

Clone the whole repository to your local directory

Run docker

```
docker run --runtime=nvidia --rm -it --mount type=bind,src=[your local directory],dst=[your destination direcoty in docker] tensorflow/tensorflow:2.0.0a0-gpu-py3 /bin/bash
```

Once in docker, install pyyaml

```
pip install pyyaml
```

Change directory to your destination direcoty in docker
```
cd [your destination direcoty in docker]
```

Run the model.py, the current setting will generate MRR score of a predefined test with saved model and data. The output is saved into temp.txt. To change to train mode, modify the test flag in main function of model.py. To test with different beam size or different  test sequence number, change the parameters in main function also
```
python -u model.py ../data/Train-BPE-txt -v ../data/Valid-BPE-txt | tee temp.txt
```
