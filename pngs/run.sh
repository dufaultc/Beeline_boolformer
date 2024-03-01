# python BLRunner.py --config config-files/config-gsd.yaml
# python BLEvaluator.py --config config-files/config-gsd.yaml --auc -t
# python BLPlotter.py --config config-files/config-gsd.yaml --auprc --auroc
# python BLRunner.py --config config-files/config-mcad.yaml
# python BLEvaluator.py --config config-files/config-mcad.yaml --auc -t
# python BLPlotter.py --config config-files/config-mcad.yaml --auprc --auroc
# python BLRunner.py --config config-files/config-vsc.yaml
# python BLEvaluator.py --config config-files/config-vsc.yaml --auc -t
# python BLPlotter.py --config config-files/config-vsc.yaml --auprc --auroc
# python BLRunner.py --config config-files/config-hsc.yaml
# python BLEvaluator.py --config config-files/config-hsc.yaml --auc -t
# python BLPlotter.py --config config-files/config-hsc.yaml --auprc --auroc
# python BLRunner.py --config config-files/config-dyn-li.yaml
# python BLEvaluator.py --config config-files/config-dyn-li.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-li.yaml --auprc --auroc -e -v
# python BLRunner.py --config config-files/config-dyn-bf.yaml
# python BLEvaluator.py --config config-files/config-dyn-bf.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-bf.yaml --auprc --auroc -e -v
# python BLRunner.py --config config-files/config-dyn-ll.yaml
# python BLEvaluator.py --config config-files/config-dyn-ll.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-ll.yaml --auprc --auroc -e -v
# python BLRunner.py --config config-files/config-dyn-bfc.yaml
# python BLEvaluator.py --config config-files/config-dyn-bfc.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-bfc.yaml --auprc --auroc -e -v
# python BLRunner.py --config config-files/config-dyn-tf.yaml
# python BLEvaluator.py --config config-files/config-dyn-tf.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-tf.yaml --auprc --auroc -e -v
# python BLRunner.py --config config-files/config-dyn-cy.yaml
# python BLEvaluator.py --config config-files/config-dyn-cy.yaml --auc -j -t -r -e -p
python BLPlotter-Synthetic.py --config config-files/config-dyn-cy.yaml --auprc --auroc -e -v