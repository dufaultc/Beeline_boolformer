import os
import re
import pandas as pd
from pathlib import Path
import numpy as np
import numpy as np
import argparse
from collections import defaultdict
import os
import time
import tqdm
import pandas as pd
from boolformer import load_boolformer

import numpy as np
import torch
from sklearn.cluster import KMeans


# Boolformer code adaption of https://github.com/sdascoli/boolformer/blob/main/scripts/evaluate_on_grn.py

#The following two methods kmeans_cluster and get_min_avg_cluster were created with the help of generative AI tools
def kmeans_cluster(row_values, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(row_values.reshape(-1, 1))
    return kmeans.labels_
def get_min_avg_cluster(row, clusters):
    cluster_avg_values = {}
    
    for cluster_label in clusters.unique():
        cluster_avg_values[cluster_label] = np.mean(row[clusters == cluster_label])

    min_avg_cluster = min(cluster_avg_values, key=cluster_avg_values.get)
    return min_avg_cluster

def generateInputs(RunnerObj):
    """
    Function to generate desired inputs for Boolformer.
    If the folder/files under RunnerObj.datadir exist,
    this function will not do anything.

    :param RunnerObj: An instance of the :class:`BLRun`
    """
    if not RunnerObj.inputDir.joinpath("Boolformer").exists():
        print("Input folder for Boolformer does not exist, creating input folder...")
        RunnerObj.inputDir.joinpath("Boolformer").mkdir(exist_ok=False)

    # if not RunnerObj.inputDir.joinpath("Boolformer/ExpressionData.csv").exists():
    # input data
    ExpressionData = pd.read_csv(
        RunnerObj.inputDir.joinpath(RunnerObj.exprData), header=0, index_col=0
    )
    ExpressionData = ExpressionData[:130]
    #print(ExpressionData.std(axis="columns"))
    # Convert input expression to boolean
    # If  the gene's expression value is >= it's avg. expression across cells
    # it receieves a "True", else "False"\
    #BinExpression = ExpressionData.T >= ExpressionData.mean(
    #    axis="columns"
    #  + ExpressionData.std(axis="columns")
    
    #The following three lines were created with the help of generative AI tools
    ClusteredData =  ExpressionData.apply(lambda row: kmeans_cluster(row.values, 2), axis=1, result_type='broadcast')
    min_avg_clusters = ExpressionData.apply(lambda row: get_min_avg_cluster(row, ClusteredData.loc[row.name]), axis=1)
    BinExpression = ClusteredData != min_avg_clusters[:, np.newaxis]
    
    
    #BinExpression = ExpressionData.apply(find_lowest_avg_cluster, axis=1).T
    # Write unique cells x genes output to a file 
    BinExpression.T.to_csv(RunnerObj.inputDir.joinpath("Boolformer/ExpressionData.csv"))


def run(RunnerObj):
    """
    Function to run Boolformer algorithm

    :param RunnerObj: An instance of the :class:`BLRun`
    """
    print(os.getcwd())
    inputPath = (
        "/home/cameron/repos/Beeline_boolformer"
        + str(RunnerObj.inputDir).split(str(Path.cwd()))[1]
        + "/Boolformer/ExpressionData.csv"
    )
    # make output dirs if they do not exist:
    outDir = (
        "/home/cameron/repos/Beeline_boolformer/"
        + "outputs/"
        + str(RunnerObj.inputDir).split("inputs/")[1]
        + "/Boolformer/"
    )
    os.makedirs(outDir, exist_ok=True)

    outPath = str(outDir) + "outFile.txt"
    timePath = str(outDir) + "time.txt"
    outPathDynamics = str(outDir) + "outDynamics.txt"
    boolformer_model = load_boolformer("noisy")

    boolformer_model.cuda()
    boolformer_model.embedder.params.cpu = False
    boolformer_model.eval()
    run_grn(
        boolformer_model,
        inputPath,
        outPath,
        outPathDynamics,
        timePath,
        beam_size=int(RunnerObj.params.get("beam_size", 5)),
        max_points=int(RunnerObj.params.get("max_points", 1000)),
        batch_size=int(RunnerObj.params.get("batch_size", 4)),
    )


def parseOutput(RunnerObj):
    """
    Function to parse outputs from Boolformer.

    :param RunnerObj: An instance of the :class:`BLRun`
    """
    # Quit if output directory does not exist
    outDir = "outputs/" + str(RunnerObj.inputDir).split("inputs/")[1] + "/Boolformer/"

    # Read output
    OutDF = pd.read_csv(outDir + "outFile.txt", sep="\t", header=0)

    if not Path(outDir + "outFile.txt").exists():
        print(outDir + "outFile.txt" + "does not exist, skipping...")
        return

    outFile = open(outDir + "rankedEdges.csv", "w")
    outFile.write("Gene1" + "\t" + "Gene2" + "\t" + "EdgeWeight" + "\n")

    for idx, row in OutDF.iterrows():
        outFile.write(
            "\t".join([row["TF"], row["target"], str(row["importance"])]) + "\n"
        )
    outFile.close()


def run_grn(
    model,
    inputPath,
    outPath,
    outPathDynamics,
    timePath,
    max_points=1000,
    beam_size=1,
    batch_size=4,
    sort_by="error",
):
    df = pd.read_csv(inputPath, header=0)
    df = df[df.columns[1:]]
    genes = list(df.columns)

    rows, columns = df.shape
    seriesSize = rows
    dynamic_errors, execution_times = [], []
    variable_counts = defaultdict(int)

    n_vars = len(df.columns)
    print(f"{n_vars}")
    num_datasets = n_vars
    num_batches = num_datasets // batch_size
    print(f"{num_batches}")
    start = time.time()
    pred_trees, error_arr, complexity_arr = [], [], []

    for batch in range(num_batches):
        inputs_ = df.values[None, :, :].repeat(batch_size, axis=0)

        outputs_ = np.array(
            [
                inputs_[var - batch * batch_size, 1:, var]
                for var in range(
                    batch * batch_size, min((batch + 1) * batch_size, n_vars)
                )
            ]
        )
        print(outputs_.shape)
        for var in range(batch * batch_size, min((batch + 1) * batch_size, n_vars)):
            inputs_[var - batch * batch_size, :, var] = np.random.choice(
                [0, 1],
                size=inputs_[var - batch * batch_size, :, var].shape,
                p=[0.5, 0.5],
            )
        inputs_ = inputs_[:, :-1, :]
        if max_points is not None:
            inputs_, outputs_ = inputs_[:, :max_points, :], outputs_[:, :max_points]
        print(inputs_.shape)
        pred_trees_, error_arr_, complexity_arr_ = model.fit(
            inputs_,
            outputs_,
            verbose=False,
            beam_size=beam_size,
            sort_by=sort_by,
        )
        pred_trees.extend(pred_trees_), error_arr.extend(
            error_arr_
        ), complexity_arr.extend(complexity_arr_)
    print(error_arr)
    dynamics_file = open(outPathDynamics, "w")
    structure_file = open(outPath, "w")
    time_file = open(timePath, "w")
    end = time.time()
    elapsed = end - start
    time_file.write("\n" + str(elapsed))
    time_file.close()
    structure_file.write("\t".join(["TF", "target", "importance"]))
    print(len(pred_trees))
    for idx, pred_tree in enumerate(pred_trees):
        if not pred_tree:
            continue
        pred_tree.increment_variables()
        used_variables = pred_tree.get_variables()
        for var in used_variables:
            variable_counts[var] += 1
        line = f"{genes[idx]} = {pred_tree.infix()}"
        line = line.replace("and", "&").replace("or", "||").replace("not", "!")
        line += "\n"
        dynamics_file.write(line)
        for var in used_variables:
            var_idx = int(var.split("_")[-1])
            influence = f"{genes[idx]} <- {genes[var_idx-1]}" + "\n"
            structure_file.write(
                "\n" + "\t".join([genes[var_idx - 1], genes[idx], str(1)])
            )
            # structure_file.write(influence)

    # print top 10 variables sorted by count
    print(sorted(variable_counts.items(), key=lambda x: -x[1])[:10])


