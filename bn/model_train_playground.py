import pandas as pd
from typing import Dict, Set, Tuple
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore



def network_train(evidence_binary: pd.DataFrame) -> Tuple[BayesianModel, VariableElimination]:

    black_list = [
    (diag, lab) for diag in evidence_binary.columns if diag.startswith("diagnoses_category_")
                 for lab in evidence_binary.columns if lab.startswith("lab_category_")
    ]

    
    white_list = [
    (lab, diag) for lab in evidence_binary.columns if lab.startswith("lab_category_")
                for diag in evidence_binary.columns if diag.startswith("diagnoses_category_")
    ]
    


    # Define state names for categorical variables
    state_names = {col: [-1, 0, 1] if col.startswith("lab_category_") else [0, 1]
                   for col in evidence_binary.columns}



    # Structure learning using Hill Climb Search with BIC score.
    hc = HillClimbSearch(evidence_binary)
    best_model_structure = hc.estimate(scoring_method=BicScore(evidence_binary), 
                                       max_indegree=10, 
                                       max_iter=5000,
                                       epsilon=1e-6,
                                       black_list=black_list,
                                       white_list=white_list
                                       )
    print("\nLearned Bayesian Network structure (edges):")
    print(best_model_structure.edges())

    
    model = BayesianModel(best_model_structure.edges())

    # Parameter learning using Maximum Likelihood Estimation (MLE)
    model.fit(evidence_binary, 
              estimator=MaximumLikelihoodEstimator, 
              state_names=state_names)
    

    print("\nLearned CPDs:")
    for cpd in model.get_cpds():
        print(cpd)
        print("-" * 50)

    # Create an inference engine for the learned model.
    inference = VariableElimination(model)
    return model, inference


df = pd.read_csv('data/df_train.csv')
model, inference = network_train(df)




from pyvis.network import Network
import networkx as nx

def interactive_visualization(model):
    """
    Create an interactive visualization of a BayesianModel using Pyvis.
    
    Parameters:
        model: A BayesianModel (from pgmpy) that contains your learned structure.
    
    The function will generate an HTML file ('bayesian_network.html') which you can open in your browser.
    """
    # Create a NetworkX directed graph from your model's edges.
    G = nx.DiGraph(model.edges())

    # Initialize a Pyvis network. The 'directed=True' parameter is important for Bayesian networks.
    net = Network(height="750px", width="100%", directed=True)

    # Manually add nodes with custom styling.
    for node in model.nodes():
        if node.startswith("diagnoses_category_"):
            color = "lightgreen"
        elif node.startswith("lab_category_"):
            color = "lightblue"
        else:
            color = "lightgray"
        net.add_node(node, label=node, color=color, physics=False)

    # Add edges from the Bayesian model.
    for source, target in model.edges():
        net.add_edge(source, target)

    # Specify notebook=False if you are not in a Jupyter Notebook.
    net.show("bayesian_network.html", notebook=False)

interactive_visualization(model)
