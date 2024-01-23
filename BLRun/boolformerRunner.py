import os
import random
import pandas as pd
from pathlib import Path
import numpy as np
import numpy as np
from collections import defaultdict
import os
import time
import tqdm
import pandas as pd
from boolformer import load_boolformer

import numpy as np
from sklearn.cluster import KMeans
from random import shuffle


# Boolformer code adaption of https://github.com/sdascoli/boolformer/blob/main/scripts/evaluate_on_grn.py


# The following two methods kmeans_cluster and get_min_avg_cluster were created with the help of generative AI tools
def kmeans_cluster(row_values, num_clusters, largest_zero=0.5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(row_values.reshape(-1, 1))
    cluster_labels = kmeans.labels_

    # We dont want high values in minimum cluster to be set to 0,
    # so all those above largest_zero put in their own category
    cluster_avg_values = {}
    for cluster_label in np.unique(cluster_labels):
        cluster_avg_values[cluster_label] = np.mean(
            row_values[cluster_labels == cluster_label]
        )
    min_avg_cluster = min(cluster_avg_values, key=cluster_avg_values.get)

    for i, val in enumerate(list(row_values)):
        if cluster_labels[i] == min_avg_cluster:
            if val > largest_zero:
                cluster_labels[i] == -1

    # If average of max cluster is below largest_zero, should
    # have all clusters be the same so all get binarized to zero
    max_avg_cluster = max(cluster_avg_values, key=cluster_avg_values.get)
    if cluster_avg_values[max_avg_cluster] < largest_zero:
        for i, val in enumerate(list(row_values)):
            cluster_labels[i] == 1

    return cluster_labels


def get_min_avg_cluster(row, clusters, min_to_max_cutoff=0.3):
    cluster_avg_values = {}
    for cluster_label in clusters.unique():
        cluster_avg_values[cluster_label] = np.mean(row[clusters == cluster_label])

    min_avg_cluster = min(cluster_avg_values, key=cluster_avg_values.get)
    max_avg_cluster = max(cluster_avg_values, key=cluster_avg_values.get)

    # If all cluster averages are low, every value binarized to 0
    if min_avg_cluster == max_avg_cluster:
        print("All 0")
        return max_avg_cluster

    # If the average of the minimum cluster is above
    # min_to_max_cutoff, we have no zeros, since -2 maps to no values
    if cluster_avg_values[min_avg_cluster] > (
        min_to_max_cutoff * cluster_avg_values[max_avg_cluster]
    ):
        min_avg_cluster = -2
        print("All 1")

    return min_avg_cluster


def generateInputs(RunnerObj):
    """
    Function to generate desired inputs for Boolformer.
    If the folder/files under RunnerObj.datadir exist,
    this function will overwrite them.

    :param RunnerObj: An instance of the :class:`BLRun`
    """
    if not RunnerObj.inputDir.joinpath("Boolformer").exists():
        print("Input folder for Boolformer does not exist, creating input folder...")
        RunnerObj.inputDir.joinpath("Boolformer").mkdir(exist_ok=False)

    # input gene expression data
    ExpressionData = pd.read_csv(
        RunnerObj.inputDir.joinpath(RunnerObj.exprData), header=0, index_col=0
    )

    # Boolformer can only consider up to 130 genes at a time
    max_genes = int(RunnerObj.params.get("max_genes", 130))
    if ExpressionData.shape[0] > max_genes:
        print(f"Too many genes, taking first {max_genes}")
        ExpressionData = ExpressionData[:max_genes]

    use_pseudotime = bool(RunnerObj.params.get("use_pseudotime", True))
    if use_pseudotime:
        PTData = pd.read_csv(
            RunnerObj.inputDir.joinpath(RunnerObj.cellData), header=0, index_col=0
        )
        colNames = PTData.columns
        dfs = []
        for idx, name in enumerate(colNames):
            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = PTData[colName].index[PTData[colName].notnull()]
            subPT = PTData.loc[index, :]
            subExpr = ExpressionData[index]
            newExpressionData = subExpr[subPT.sort_values([colName]).index.astype(str)]
            if bool(RunnerObj.params.get("rolling_average", False)):
                result = newExpressionData.rolling(
                    window=int(RunnerObj.params.get("rolling_window_size", 5)),
                    axis=1,
                    win_type="exponential",
                    min_periods=1,
                ).mean()
                dfs.append(result)
            else:
                dfs.append(newExpressionData)
    else:
        orderedExpressionData = ExpressionData

    exprName_save = "Boolformer/ExpressionData_precluster" + ".csv"
    orderedExpressionData = pd.concat(dfs, axis=1)
    orderedExpressionData.T.to_csv(
        RunnerObj.inputDir.joinpath(exprName_save), sep=",", header=True, index=True
    )

    exprName = "Boolformer/ExpressionData" + ".csv"
    # The following three lines were created with the help of generative AI tools
    num_clusters = int(RunnerObj.params.get("binarization_clusters", 3))
    ClusteredData = orderedExpressionData.apply(
        lambda row: kmeans_cluster(
            row.values,
            num_clusters,
            largest_zero=float(RunnerObj.params.get("largest_zero", 0.5)),
        ),
        axis=1,
        result_type="broadcast",
    )
    min_avg_clusters = orderedExpressionData.apply(
        lambda row: get_min_avg_cluster(row, ClusteredData.loc[row.name]), axis=1
    )
    BinExpression = ClusteredData != min_avg_clusters[:, np.newaxis]
    BinExpression.T.to_csv(
        RunnerObj.inputDir.joinpath(exprName), sep=",", header=True, index=True
    )


def run(RunnerObj):
    """
    Function to run Boolformer algorithm

    :param RunnerObj: An instance of the :class:`BLRun`
    """
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

    use_pseudotime = bool(RunnerObj.params.get("use_pseudotime", True))
    if use_pseudotime:
        PTData = pd.read_csv(
            RunnerObj.inputDir.joinpath(RunnerObj.cellData), header=0, index_col=0
        )
    else:
        PTData = None

    boolformer_model = load_boolformer("noisy")
    start = time.time()
    outPath = str(outDir) + "outFile" + ".txt"
    outPathDynamics = str(outDir) + "outDynamics" + ".txt"

    boolformer_model.cuda()
    boolformer_model.embedder.params.cpu = False
    boolformer_model.eval()
    run_grn(
        boolformer_model,
        inputPath,
        outPath,
        outPathDynamics,
        pseudotimeData=PTData,
        batch_size=int(RunnerObj.params.get("batch_size", 4)),
        beam_size=int(RunnerObj.params.get("beam_size", 5)),
        beam_temperature=float(RunnerObj.params.get("beam_temperature", 0.1)),
        beam_type=RunnerObj.params.get("beam_type", "search"),
        difference_emphasis=bool(RunnerObj.params.get("difference_emphasis", False)),
        max_points=int(RunnerObj.params.get("max_points", 5000)),
        pseudotime_jump=bool(RunnerObj.params.get("pseudotime_jump", True)),
        pseudotime_jump_size=float(RunnerObj.params.get("pseudotime_jump_size", 0.1)),
        repeats=int(RunnerObj.params.get("repeats", 1)),
        sample_size=int(RunnerObj.params.get("sample_size", 1000)),
        target_randomize=bool(RunnerObj.params.get("target_randomize", True)),
    )

    timePath = str(outDir) + "time.txt"
    time_file = open(timePath, "w")
    end = time.time()
    elapsed = end - start
    time_file.write("\n" + str(elapsed))
    time_file.close()


def parseOutput(RunnerObj):
    """
    Function to parse outputs from Boolformer.

    :param RunnerObj: An instance of the :class:`BLRun`
    """
    outDir = "outputs/" + str(RunnerObj.inputDir).split("inputs/")[1] + "/Boolformer/"
    outFileName = "outFile" + ".txt"

    outDF = pd.read_csv(outDir + outFileName, sep="\t", header=0)
    FinalDF = outDF[
        outDF["importance"]
        == outDF.groupby(["TF", "target"])["importance"].transform("max")
    ]
    FinalDF.drop_duplicates(inplace=True)

    outFile = open(outDir + "rankedEdges.csv", "w")
    outFile.write("Gene1" + "\t" + "Gene2" + "\t" + "EdgeWeight" + "\n")
    FinalDF.sort_values("importance", inplace=True, ascending=False)

    for idx, row in FinalDF.iterrows():
        outFile.write(
            "\t".join([row["TF"], row["target"], str(row["importance"])]) + "\n"
        )
    outFile.close()


def run_grn(
    model,
    inputPath,
    outPath,
    outPathDynamics,
    pseudotimeData,
    batch_size=4,
    beam_size=5,
    beam_temperature=0.1,
    beam_type="search",
    difference_emphasis=False,
    max_points=5000,
    pseudotime_jump=True,
    pseudotime_jump_size=0.1,
    repeats=1,
    target_randomize=True,
    sample_size=1000,
    sort_by="error",
):
    df = pd.read_csv(inputPath, header=0)
    df = df[df.columns[1:]]
    genes = list(df.columns)

    rows, columns = df.shape
    variable_counts = defaultdict(int)

    n_vars = len(df.columns)
    num_datasets = n_vars
    num_batches = num_datasets // batch_size

    if pseudotimeData is not None:
        # x = random.sample(range(df.shape[0]), 50)
        # x.sort()
        # df = df.iloc[x, :]
        rows, columns = df.shape

        colNames = pseudotimeData.columns
        pseudotimes_list = []
        for idx, name in enumerate(colNames):
            # Select cells belonging to each pseudotime trajectory
            colName = colNames[idx]
            index = pseudotimeData[colName].index[pseudotimeData[colName].notnull()]

            subPT = pseudotimeData.loc[index, :]
            pseudotimes_list.extend(list(subPT.sort_values([colName])[colName].values))

        # pseudotimes_list = [pseudotimes_list[gah] for gah in x]
        next_lists = [[] for _ in range(n_vars)]
        for gene in range(n_vars):
            for i in range(rows - 1):
                start = i + 1
                for j in range(start, rows):
                    if pseudotime_jump:
                        if (
                            pseudotimes_list[j] - pseudotimes_list[i]
                        ) >= pseudotime_jump_size:
                            next_lists[gene].append(j)
                            break
                    else:
                        if pseudotimes_list[i] < pseudotimes_list[j]:
                            next_lists[gene].append(j)
                            break
                    if (pseudotimes_list[i] > pseudotimes_list[j]) or (j == rows - 1):
                        next_lists[gene].append(i)
                        break

        """
        For every cell, for a given target gene
        
        set the output for the cell to the target value at 
            - The next change in target value if within 0.1 psuedotime but beyond 0.005
            - The current target value otherwise
        """
        if difference_emphasis:
            for gene in range(n_vars):
                for i in range(rows - 1):
                    for j in range(i + 1, rows):
                        if (
                            pseudotimes_list[j]
                            > pseudotimes_list[i] + pseudotime_jump_size * 10
                        ) or (pseudotimes_list[j] < pseudotimes_list[i]):
                            next_lists[gene].append(i + 1)
                            break
                        if df.values[None, i, gene] != df.values[None, j, gene] and (
                            pseudotimes_list[j]
                            > pseudotimes_list[i] + pseudotime_jump_size
                        ):
                            next_lists[gene].append(j)
                            break
                        if j == rows - 1:
                            next_lists[gene].append(i + 1)
                            break

            df = pd.concat([df.iloc[:-1], df])

    pred_trees_many = []
    for i in tqdm.tqdm(range(repeats)):
        pred_trees, error_arr, complexity_arr = [], [], []
        for batch in range(num_batches):
            inputs_ = df.values[None, :, :].repeat(batch_size, axis=0)

            if pseudotimeData is not None:
                outputs_ = np.array(
                    [
                        inputs_[var - batch * batch_size, next_lists[var], var]
                        for var in range(
                            batch * batch_size, min((batch + 1) * batch_size, n_vars)
                        )
                    ]
                )
            else:
                outputs_ = np.array(
                    [
                        inputs_[var - batch * batch_size, 1:, var]
                        for var in range(
                            batch * batch_size, min((batch + 1) * batch_size, n_vars)
                        )
                    ]
                )

            if target_randomize:
                for var in range(
                    batch * batch_size, min((batch + 1) * batch_size, n_vars)
                ):
                    inputs_[var - batch * batch_size, :, var] = np.random.choice(
                        [0, 1],
                        size=inputs_[var - batch * batch_size, :, var].shape,
                        p=[0.5, 0.5],
                    )
            else:
                inputs_ = np.array(
                    [
                        np.squeeze(
                            df.drop(df.columns[[num]], axis=1).values[None, :, :],
                            axis=0,
                        )
                        for num in range(
                            batch * batch_size, min((batch + 1) * batch_size, n_vars)
                        )
                    ]
                )

            inputs_ = inputs_[:, :-1, :]

            if max_points is not None:
                inputs_, outputs_ = inputs_[:, :max_points, :], outputs_[:, :max_points]
            x = random.sample(range(inputs_.shape[1]), sample_size)
            pred_trees_, error_arr_, complexity_arr_ = model.fit(
                inputs_[:, x, :],
                outputs_[:, x],
                verbose=False,
                beam_size=beam_size,
                beam_temperature=beam_temperature,
                beam_type=beam_type,
                sort_by=sort_by,
            )

            pred_trees.extend(pred_trees_), error_arr.extend(
                error_arr_
            ), complexity_arr.extend(complexity_arr_)
        pred_trees_many.append(pred_trees)

    print(error_arr)
    dynamics_file = open(outPathDynamics, "w")
    structure_file = open(outPath, "w")
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
        used_variables = list(used_variables)
        used_variables.sort()
        for var in used_variables:
            index = int(var.split("_")[1]) - 1
            if target_randomize == False and index >= idx:
                index = index + 1
            if len(genes) <= index:
                continue
            line = line.replace(var, genes[index])
        line += "\n"
        pred_tree.decrement_variables()
        dynamics_file.write(line)

    count_dict = {gene: {gene_inner: 0 for gene_inner in genes} for gene in genes}
    for i in range(repeats):
        pred_trees = pred_trees_many[i]
        for idx, pred_tree in enumerate(pred_trees):
            if not pred_tree:
                continue
            pred_tree.increment_variables()
            used_variables = pred_tree.get_variables()
            used_variables = list(used_variables)
            used_variables.sort()
            for var in used_variables:
                index = int(var.split("_")[1]) - 1
                if target_randomize == False and index >= idx:
                    index = index + 1
                if len(genes) <= index:
                    continue
                count_dict[genes[idx]][genes[index]] += 1
    print(count_dict)
    for gene in genes:
        for gene_inner in genes:
            structure_file.write(
                "\n"
                + "\t".join(
                    [gene_inner, gene, str(count_dict[gene][gene_inner] / repeats)]
                )
            )

    structure_file.close()

    # print top 10 variables sorted by count
    print(sorted(variable_counts.items(), key=lambda x: -x[1])[:10])
