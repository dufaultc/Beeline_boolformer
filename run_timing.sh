# python BLRunner.py --config config-files/config-mcad.yaml
# python BLEvaluator.py --config config-files/config-mcad.yaml --auc -j -t -r -e -p
# python BLPlotter-Curated.py --config config-files/config-mcad.yaml --auprc --auroc -e -v
# python BLRunner.py --config config-files/config-vsc.yaml
# python BLEvaluator.py --config config-files/config-vsc.yaml --auc -j -t -r -e -p
# python BLPlotter-Curated.py --config config-files/config-vsc.yaml --auprc --auroc -e -v
# python BLRunner.py --config config-files/config-hsc.yaml
# python BLEvaluator.py --config config-files/config-hsc.yaml --auc -j -t -r -e -p
# python BLPlotter-Curated.py --config config-files/config-hsc.yaml --auprc --auroc -e -v
# python BLRunner.py --config config-files/config-gsd.yaml
# python BLEvaluator.py --config config-files/config-gsd.yaml --auc -j -t -r -e -p
# python BLPlotter-Curated.py --config config-files/config-gsd.yaml --auprc --auroc -e -v


python BLRunner.py --config config-files/config-dyn-li-timing.yaml
python BLEvaluator.py --config config-files/config-dyn-li-timing.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-li-timing.yaml --auprc --auroc -e -v
python BLRunner.py --config config-files/config-dyn-bf-timing.yaml
python BLEvaluator.py --config config-files/config-dyn-bf-timing.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-bf-timing.yaml --auprc --auroc -e -v
python BLRunner.py --config config-files/config-dyn-ll-timing.yaml
python BLEvaluator.py --config config-files/config-dyn-ll-timing.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-ll-timing.yaml --auprc --auroc -e -v
python BLRunner.py --config config-files/config-dyn-bfc-timing.yaml
python BLEvaluator.py --config config-files/config-dyn-bfc-timing.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-bfc-timing.yaml --auprc --auroc -e -v
python BLRunner.py --config config-files/config-dyn-tf-timing.yaml
python BLEvaluator.py --config config-files/config-dyn-tf-timing.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-tf-timing.yaml --auprc --auroc -e -v
python BLRunner.py --config config-files/config-dyn-cy-timing.yaml
python BLEvaluator.py --config config-files/config-dyn-cy-timing.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-cy-timing.yaml --auprc --auroc -e -v