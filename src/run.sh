echo "Node_classification results: "
python train.py --dataset=cora --hops=20 --dropout=0.2 --lr=2e-1 --weight_decay=5e-3
python train.py --dataset=citeseer --hops=15 --dropout=0.1 --lr=1e-1 --weight_decay=5e-2
python train.py --dataset=pubmed --hops=20  --dropout=0.5  --lr=2e-1  --weight_decay=5e-5