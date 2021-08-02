## An Experimental Evaluation of Using Deep Convolution on Graph-Structured Data



### Requirements

Environments: Xeon Gold 5120 (CPU), 384GB(RAM), TITAN RTX (GPU), Ubuntu 16.04 (OS).

The PyTorch version we use is torch 1.7.1+cu110. Please refer to the official website -- https://pytorch.org/get-started/locally/ -- for the detailed installation instructions.

To install all the requirements:

```setup
pip install -r requirements.txt
```



### Training

To reproduce the results on the Cora, Citeseer, Pubmed dataset, please run this command:

```train
bash ./src/run.sh
```

 

### Node Classification Results:

![node_classification_on_citation_networks](node_classification_on_citation_networks.png)
