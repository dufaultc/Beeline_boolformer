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
import random


# Boolformer code adaption of https://github.com/sdascoli/boolformer/blob/main/scripts/evaluate_on_grn.py


# The following two methods kmeans_cluster and get_min_avg_cluster were created with the help of generative AI tools
def kmeans_cluster(row_values, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(row_values.reshape(-1, 1))
    cluster_labels = kmeans.labels_
    
    
    cluster_avg_values = {}

    for cluster_label in np.unique(cluster_labels):
        cluster_avg_values[cluster_label] = np.mean(row_values[cluster_labels == cluster_label])

    min_avg_cluster = min(cluster_avg_values, key=cluster_avg_values.get)
    max_avg_cluster = max(cluster_avg_values, key=cluster_avg_values.get)
    if cluster_avg_values[min_avg_cluster] > (
        0.20 * cluster_avg_values[max_avg_cluster]
    ):
        print("cluster adjust boolformer correct")
        if (min(row_values) < 0.45):
            min_avg_cluster = -1
            cluster_labels = [-1 if val < 0.45 else 1 for val in list(row_values)]
    
    return cluster_labels


def get_min_avg_cluster(row, clusters):
    cluster_avg_values = {}

    for cluster_label in clusters.unique():
        cluster_avg_values[cluster_label] = np.mean(row[clusters == cluster_label])

    min_avg_cluster = min(cluster_avg_values, key=cluster_avg_values.get)
    max_avg_cluster = max(cluster_avg_values, key=cluster_avg_values.get)
    if cluster_avg_values[min_avg_cluster] > (
        0.30 * cluster_avg_values[max_avg_cluster]
    ):
        print("cluster adjust boolformer shouldnt happen")
        min_avg_cluster = -1
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
    # ExpressionData = ExpressionData[:130]

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
        #result = newExpressionData.rolling(window=5, axis=1, min_periods=1).mean()

        dfs.append(newExpressionData)

    exprName = "Boolformer/ExpressionData" + ".csv"        
    
    exprName_save = "Boolformer/ExpressionData_real" + ".csv"
    
    
    orderedExpressionData = pd.concat(dfs, axis = 1)
    orderedExpressionData.T.to_csv(
        RunnerObj.inputDir.joinpath(exprName_save), sep=",", header=True, index=True
    )
    
    
    # The following three lines were created with the help of generative AI tools
    ClusteredData = orderedExpressionData.apply(
        lambda row: kmeans_cluster(row.values, 3), axis=1, result_type="broadcast"
    )
    min_avg_clusters = orderedExpressionData.apply(
        lambda row: get_min_avg_cluster(row, ClusteredData.loc[row.name]), axis=1
    )
    print()
    BinExpression = ClusteredData != min_avg_clusters[:, np.newaxis]
    #BinExpression = orderedExpressionData >= 0.50
    #BinExpression.drop_duplicates(inplace=True)


    BinExpression.T.to_csv(
        RunnerObj.inputDir.joinpath(exprName), sep=",", header=True, index=True
    )
        # BinExpression.T.to_csv(RunnerObj.inputDir.joinpath("Boolformer/ExpressionData.csv"))

    # BinExpression = ExpressionData.apply(find_lowest_avg_cluster, axis=1).T
    # Write unique cells x genes output to a file


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
    PTData = pd.read_csv(
        RunnerObj.inputDir.joinpath(RunnerObj.cellData), header=0, index_col=0
    )

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
        PTData,
        beam_size=int(RunnerObj.params.get("beam_size", 5)),
        max_points=int(RunnerObj.params.get("max_points", 1000)),
        batch_size=int(RunnerObj.params.get("batch_size", 4)),
        repeats=int(RunnerObj.params.get("repeats", 1))
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
    # Quit if output directory does not exist
    outDir = "outputs/" + str(RunnerObj.inputDir).split("inputs/")[1] + "/Boolformer/"

    PTData = pd.read_csv(
        RunnerObj.inputDir.joinpath(RunnerObj.cellData), header=0, index_col=0
    )

    colNames = PTData.columns

    outFileName = "outFile" + ".txt"

    outDF = pd.read_csv(outDir + outFileName, sep="\t", header=0)
    FinalDF = outDF[
        outDF["importance"]
        == outDF.groupby(["TF", "target"])["importance"].transform("max")
    ]
    FinalDF.drop_duplicates(inplace=True)
    # Read output

    outFile = open(outDir + "rankedEdges.csv", "w")
    outFile.write("Gene1" + "\t" + "Gene2" + "\t" + "EdgeWeight" + "\n")

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
    max_points=1000,
    beam_size=1,
    batch_size=4,
    repeats=1,
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
    # print(f"{n_vars}")
    num_datasets = n_vars
    num_batches = num_datasets // batch_size
    # print(f"{num_batches}")

    #pred_trees, error_arr, complexity_arr = [], [], []
    
    colNames = pseudotimeData.columns
    pseudotimes_list = []
    for idx, name in enumerate(colNames):
        # Select cells belonging to each pseudotime trajectory
        colName = colNames[idx]
        index = pseudotimeData[colName].index[pseudotimeData[colName].notnull()]
        
        subPT = pseudotimeData.loc[index, :]
        pseudotimes_list.extend(list(subPT.sort_values([colName])[colName].values))

    #print(subPT.sort_values([colName]))
    #print(pseudotimes_list)    
    
    #next_lists = [[] for _ in range(n_vars)]  
    next_list = []
    #for gene in range(n_vars):
    for i in range(rows-1):
        #val = random.uniform(0, 1)
        if i+1 >= rows:
            start = i+1
        else:
            start = i+1
        #start = i + 1    
        for j in range(start,rows):
            if pseudotimes_list[i] < pseudotimes_list[j]:
                #next_lists[gene].append(j)
                next_list.append(j)
                break
            if (pseudotimes_list[i] > pseudotimes_list[j]) or (j == rows-1):
                next_list.append(i)
                break  
    # for i in range(rows-1):
    #     skip = random.randint(0, 20)
    #     if i+skip >= rows:
    #         start = i+1
    #     else:
    #         start = i+skip
    #     for j in range(start,rows):
    #         if pseudotimes_list[i] < pseudotimes_list[j]:
    #             #next_lists[gene].append(j)
    #             next_list.append(j)
    #             break
    #         if (pseudotimes_list[i] > pseudotimes_list[j]) or (j == rows-1):
    #             next_list.append(i)
    #             break              


    '''
    For every cell, for a given target gene
    
    set the output for the cell to the target value at 
        - The next change in target value if within 0.1 psuedotime but beyond 0.005
        - The current target value otherwise
    
    

    for gene in range(n_vars):
        for i in range(rows-1):      
            for j in range(i+1,rows):
                if (pseudotimes_list[j] > pseudotimes_list[i] + 0.05) or (pseudotimes_list[j] < pseudotimes_list[i]):
                    next_lists[gene].append(i+1)
                    break                
                if df.values[None, i, gene] != df.values[None, j, gene] and (pseudotimes_list[j] > pseudotimes_list[i] + 0.01):
                    next_lists[gene].append(j)
                    break                
                if j == rows-1:
                    next_lists[gene].append(i+1)                    
                    break
    '''
         
    # last_j= -1
    # last_change = []
    # for i in range(rows-1):               
    #     for j in range(i+1,rows):
    #         if (pseudotimes_list[i] > pseudotimes_list[j]) or (j == rows-1):
    #             next_list.append(i)  
    #             break      
    #         elif not (np.array_equal(df.values[None, i, :], df.values[None, j, :])):     
    #             difference = np.where(df.values[None, i, :] != df.values[None, j, :])[1].tolist()                 
    #             if i == last_j:                    
    #                 if difference == last_change:                    
    #                     continue   
    #             elif i < last_j:
    #                 next_list.append(last_j)
    #                 break                 
    #             next_list.append(j)
    #             last_change = difference
    #             last_j = j
    #             break       
    

            
            
    #for i,change in enumerate(next_lists[2][1999:2999]):
    #    print(i,change)
        
      

    #df_new = pd.concat([df.iloc[:-1],df])
    #print(df_new.shape)
    pred_trees_many = []
    for i in range(repeats):
        pred_trees, error_arr, complexity_arr = [], [], []
        for batch in range(num_batches):
            
            #inputs_ = df_new.values[None, :, :].repeat(batch_size, axis=0)
            inputs_ = df.values[None, :, :].repeat(batch_size, axis=0)
            #print(inputs_.shape)
            outputs_ = np.array(
                [
                    #inputs_[var - batch * batch_size, next_lists[var], var]
                    inputs_[var - batch * batch_size, next_list, var]
                    for var in range(
                        batch * batch_size, min((batch + 1) * batch_size, n_vars)
                    )
                ]
            )
            # print(outputs_.shape

            for var in range(batch * batch_size, min((batch + 1) * batch_size, n_vars)):
                inputs_[var - batch * batch_size, :, var] = np.random.choice(
                    [0, 1],
                    size=inputs_[var - batch * batch_size, :, var].shape,
                    p=[0.5, 0.5],
                )
            inputs_ = inputs_[:, :-1, :]

            
            if max_points is not None:
                inputs_, outputs_ = inputs_[:, :max_points, :], outputs_[:, :max_points]
            # print(inputs_.shape)
            pred_trees_, error_arr_, complexity_arr_ = model.fit(
                inputs_,
                outputs_,
                verbose=False,
                beam_size=beam_size,
                # beam_temperature=0.0005,
                beam_temperature=0.5,
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
        #print( list(used_variables))
        #print( list(genes))
        used_variables = list(used_variables)
        used_variables.sort()
        for var in used_variables:
            index = int(var.split("_")[1]) - 1
            if len(genes) <= index:
                continue 
            line = line.replace(var, genes[index])
        line += "\n"
        pred_tree.decrement_variables()
        dynamics_file.write(line)
    
    count_dict = {gene : {gene_inner : 0 for gene_inner in genes} for gene in genes}
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
                if len(genes) <= index:
                    continue                 
                count_dict[genes[idx]][genes[index]] += 1
    print(count_dict)
    for gene in genes:
        for gene_inner in genes:
            structure_file.write(
                "\n" + "\t".join([gene_inner, gene, str(count_dict[gene][gene_inner]/repeats)])
            )
                
    
            # structure_file.write(influence)

    # print top 10 variables sorted by count
    print(sorted(variable_counts.items(), key=lambda x: -x[1])[:10])
