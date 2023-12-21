python BLRunner.py --config config-files/config-gsd.yaml
python BLEvaluator.py --config config-files/config-gsd.yaml --auc -t
python BLPlotter.py --config config-files/config-gsd.yaml --auprc --auroc
python BLRunner.py --config config-files/config-mcad.yaml
python BLEvaluator.py --config config-files/config-mcad.yaml --auc -t
python BLPlotter.py --config config-files/config-mcad.yaml --auprc --auroc
#python BLRunner.py --config config-files/config-dyn-li.yaml
#python BLEvaluator.py --config config-files/config-dyn-li.yaml --auc -t
#python BLPlotter.py --config config-files/config-dyn-li.yaml --auprc --auroc
#python BLRunner.py --config config-files/config-dyn-bf.yaml
#python BLEvaluator.py --config config-files/config-dyn-bf.yaml --auc -t
#python BLPlotter.py --config config-files/config-dyn-bf.yaml --auprc --auroc