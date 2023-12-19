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

def generateInputs(RunnerObj):
    '''
    Function to generate desired inputs for Boolformer.
    If the folder/files under RunnerObj.datadir exist, 
    this function will not do anything.

    :param RunnerObj: An instance of the :class:`BLRun`
    '''
    if not RunnerObj.inputDir.joinpath("Boolformer").exists():
        print("Input folder for Boolformer does not exist, creating input folder...")
        RunnerObj.inputDir.joinpath("Boolformer").mkdir(exist_ok = False)
        
    #if not RunnerObj.inputDir.joinpath("Boolformer/ExpressionData.csv").exists():
        # input data
    ExpressionData = pd.read_csv(RunnerObj.inputDir.joinpath(RunnerObj.exprData),
                                    header = 0, index_col = 0)

    # Convert input expression to boolean
    # If  the gene's expression value is >= it's avg. expression across cells
    # it receieves a "True", else "False"
    BinExpression = ExpressionData.T >= ExpressionData.mean(axis = 'columns')
    BinExpression.drop_duplicates(inplace= True)
    # Write unique cells x genes output to a file
    BinExpression.to_csv(RunnerObj.inputDir.joinpath("Boolformer/ExpressionData.csv")) 
    
def run(RunnerObj):
    '''
    Function to run Boolformer algorithm

    :param RunnerObj: An instance of the :class:`BLRun`
    '''
    print(os.getcwd())
    inputPath = "/home/cameron/repos/Beeline_boolformer" + str(RunnerObj.inputDir).split(str(Path.cwd()))[1] + \
                    "/Boolformer/ExpressionData.csv"
    # make output dirs if they do not exist:
    outDir =  "/home/cameron/repos/Beeline_boolformer/" + "outputs/"+str(RunnerObj.inputDir).split("inputs/")[1]+"/Boolformer/"
    os.makedirs(outDir, exist_ok = True)
    
    outPath = str(outDir) + 'outFile.txt'
    timePath = str(outDir) + 'time.txt'
    outPathDynamics = str(outDir) + 'outDynamics.txt'
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
        beam_size=5,
    )    



def parseOutput(RunnerObj):
    '''
    Function to parse outputs from GENIE3.

    :param RunnerObj: An instance of the :class:`BLRun`
    '''
    # Quit if output directory does not exist
    outDir = "outputs/"+str(RunnerObj.inputDir).split("inputs/")[1]+"/Boolformer/"

        
    # Read output
    OutDF = pd.read_csv(outDir+'outFile.txt', sep = '\t', header = 0)
    
    if not Path(outDir+'outFile.txt').exists():
        print(outDir+'outFile.txt'+'does not exist, skipping...')
        return
    
    outFile = open(outDir + 'rankedEdges.csv','w')
    outFile.write('Gene1'+'\t'+'Gene2'+'\t'+'EdgeWeight'+'\n')

    for idx, row in OutDF.iterrows():
        outFile.write('\t'.join([row['TF'],row['target'],str(row['importance'])])+'\n')
    outFile.close()
    

def run_grn(
    model,
    inputPath,
    outPath,
    outPathDynamics,
    timePath,
    max_points = 1000,
    verbose=False,
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

    inputs = df.values[None, :, :].repeat(n_vars, axis=0)
    outputs = np.array([inputs[var, 1:, var] for var in range(n_vars)])
    # inputs = np.array([np.concatenate((inputs[var,:,:var],inputs[var,:,var+1:]), axis=-1) for var in range(n_vars)])
    for var in range(n_vars):
        inputs[var, :, var] = np.random.choice(
            [0, 1], size=inputs[var, :, var].shape, p=[0.5, 0.5]
        )
    inputs = inputs[:, :-1, :]
    if max_points is not None:
        # indices = np.random.choice(range(inputs.shape[1]), max_points, replace=False)
        # inputs, outputs = inputs[:,indices,:], outputs[:,indices]
        inputs, outputs = inputs[:, :max_points, :], outputs[:, :max_points]

    num_datasets = len(inputs)
    num_batches = num_datasets // batch_size

    start = time.time()
    pred_trees, error_arr, complexity_arr = [], [], []
    for batch in range(num_batches):
        inputs_, outputs_ = (
            inputs[batch * batch_size : (batch + 1) * batch_size],
            outputs[batch * batch_size : (batch + 1) * batch_size],
        )

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
    end = time.time()
    elapsed = end - start

    dynamics_file = open(outPathDynamics, "w")
    structure_file = open(outPath, "w")
    time_file = open(timePath, "w")
    time_file.write("\n"+str(elapsed))
    time_file.close()
    structure_file.write('\t'.join(["TF", "target", "importance"]))
    for idx, pred_tree in enumerate(pred_trees):
        if not pred_tree:
            continue
        pred_tree.increment_variables()
        used_variables = pred_tree.get_variables()
        for var in used_variables:
            variable_counts[var] += 1
        line = f"{genes[idx]} = {pred_tree.infix()}"
        line = (
            line.replace("and", "&")
            .replace("or", "||")
            .replace("not", "!")
        )
        line += "\n"
        dynamics_file.write(line)
        for var in used_variables:
            var_idx = int(var.split("_")[-1])
            influence = f"{genes[idx]} <- {genes[var_idx-1]}" + "\n"
            structure_file.write('\n'+'\t'.join([genes[var_idx-1], genes[idx], str(1)]))
            #structure_file.write(influence)


    # print top 10 variables sorted by count
    print(sorted(variable_counts.items(), key=lambda x: -x[1])[:10])


def getTargetGenesEvalExpressions(bool_expressions):
    target_genes = []
    eval_expressions = []
    for k in range(0, len(bool_expressions)):
        expr = bool_expressions[k]
        gene_num = int(re.search(r"\d+", expr[: expr.find(" = ")]).group())
        eval_expr = expr[expr.find("= ") + 2 :]
        target_genes.append(gene_num)
        eval_expressions.append(eval_expr)
    return target_genes, eval_expressions


def getBooleanExpressions(model_path):
    bool_expressions = []
    with open(model_path) as f:
        bool_expressions = [
            line.replace("!", " not ")
            .replace("&", " and ")
            .replace("||", " or ")
            .strip()
            for line in f
        ]
    return bool_expressions


def evalBooleanModel(model_path, test_series):
    rows, columns = test_series.shape
    simulations = test_series.iloc[[0]].copy()  # set initial states
    bool_expressions = getBooleanExpressions(model_path)
    target_genes, eval_expressions = getTargetGenesEvalExpressions(bool_expressions)

    # intialize genes to false
    for k in range(0, columns):
        gene_num = k + 1
        exec("Gene" + str(gene_num) + " = False", globals())

    for time_stamp in range(1, rows):
        # dynamically allocate variables
        for k in range(0, len(target_genes)):
            gene_num = target_genes[k]
            exec(
                "Gene"
                + str(gene_num)
                + " = "
                + str(simulations.iat[time_stamp - 1, gene_num - 1])
            )

        # initialize simulation to false
        ex_row = [0] * columns
        # evaluate all expression
        for k in range(0, len(bool_expressions)):
            gene_num = target_genes[k]
            eval_expr = eval_expressions[k]
            # print(eval_expr, eval(eval_expr))
            ex_row[gene_num - 1] = int(eval(eval_expr))
        simulations = simulations._append([ex_row], ignore_index=True)

    errors = simulations.sub(test_series)
    return np.absolute(errors.to_numpy()).sum(), simulations, test_series