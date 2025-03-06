from typing import Dict, List, Union, Tuple

from pyvis.network import Network
import networkx as nx
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork



def network_train(evidence_binary: pd.DataFrame) -> Tuple[BayesianNetwork, VariableElimination]:
    """Train a Bayesian network from binary evidence data.

    This function performs structure learning using Hill Climb Search with the Bayesian Information
    Criterion (BIC) scoring method. It then estimates the conditional probability distributions (CPDs)
    using Maximum Likelihood Estimation (MLE) and constructs a Variable Elimination inference engine.

    Args:
        evidence_binary (pd.DataFrame): A one-hot encoded DataFrame where each column represents an
            observed variable (e.g., ICD code) and each row represents a visit.

    Returns:
        model (BayesianModel): The Bayesian network with learned structure and parameters.
        inference (VariableElimination): An inference engine for performing probabilistic queries.
    """
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
                                       #white_list=white_list
                                       )
    print("\nLearned Bayesian Network structure (edges):")
    print(best_model_structure.edges())

    
    model = BayesianNetwork(best_model_structure.edges())

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



def network_visualisation(model: BayesianNetwork) -> None:
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



def entropy(prob_dist: np.ndarray) -> float:
    """
    Compute the Shannon entropy (in bits) of a 1D probability distribution.
    The distribution must sum to 1; zero-probability entries are ignored.
    """
    # Use only p>0 to avoid log(0)
    return -np.sum(p * np.log2(p) for p in prob_dist if p > 0)



def expected_entropy_for_node(
    candidate_node: str,
    current_evidence: dict,
    target_nodes: List[str],
    inference: VariableElimination
) -> float:

    # Query the marginal for the candidate node given current evidence
    marginal = inference.query(variables=[candidate_node], evidence=current_evidence)
    marginal_probs = marginal.values
    candidate_states = marginal.state_names[candidate_node]  # actual labels
    
    exp_entropy = 0.0
    
    # Iterate over each possible state (label) in candidate_node
    for i, state_label in enumerate(candidate_states):
        p_state = marginal_probs[i]
        
        # Update evidence with this candidate node's state
        new_evidence = dict(current_evidence)
        new_evidence[candidate_node] = state_label
        
        # Sum entropies of each target node's posterior
        sum_entropies = 0.0
        for t_node in target_nodes:
            posterior = inference.query(variables=[t_node], evidence=new_evidence)
            sum_entropies += entropy(posterior.values)
        
        # Weight by probability of candidate_node = state_label
        exp_entropy += p_state * sum_entropies
    
    return exp_entropy

def perform_value_of_information(
    inference: VariableElimination,
    model: BayesianNetwork,  # UPDATED parameter type
    target_nodes: Union[str, List[str]],
    current_evidence: dict
) -> Dict[str, float]:

    # Ensure target_nodes is a list, even if a single string is passed.
    if isinstance(target_nodes, str):
        target_nodes = [target_nodes]

    # Compute baseline sum of entropies for the target nodes
    baseline_entropy_sum = 0.0
    for t_node in target_nodes:
        posterior_target = inference.query(variables=[t_node], evidence=current_evidence)
        baseline_entropy_sum += entropy(posterior_target.values)

    print(f"\nBaseline Entropy (sum across {target_nodes}): {baseline_entropy_sum:.4f} bits")
    
    # Identify candidate nodes = all nodes - already observed - target_nodes
    all_nodes = set(model.nodes())
    observed_nodes = set(current_evidence.keys())
    excluded_nodes = observed_nodes.union(target_nodes)
    candidate_nodes = all_nodes - excluded_nodes
    
    info_gain = {}
    for candidate in candidate_nodes:
        # Expected sum of entropies if we observe 'candidate'
        exp_ent = expected_entropy_for_node(candidate, current_evidence, target_nodes, inference)
        gain = baseline_entropy_sum - exp_ent
        info_gain[candidate] = gain
        
        print(f"Candidate: {candidate:20s} | Expected Entropy: {exp_ent:.4f} | Info Gain: {gain:.4f}")
    
    return info_gain



evidence_binary = pd.read_csv('data/df_train.csv')
model, inference = network_train(evidence_binary)
network_visualisation(model)


current_evidence = {"lab_category_platelets": 1, "lab_category_bun": 1}
target_nodes = ["diagnoses_category_gi", "diagnoses_category_sepsis", "diagnoses_category_pneumonia", "diagnoses_category_aci", "diagnoses_category_chf"]
#target_nodes = ["diagnoses_category_gi"]


results = perform_value_of_information(inference, model, target_nodes, current_evidence)
if results:
    best_candidate = max(results, key=results.get)
    print(f"\nBest node to observe next: {best_candidate} (Gain = {results[best_candidate]:.4f} bits)")
else:
    print("\nNo candidate nodes available for additional observation.")



# Find the most likely outcome (argmax of probability values)
query_result = inference.query(variables=target_nodes, evidence=current_evidence)

top_indices = np.argsort(query_result.values.flatten())[-3:][::-1]
state_combinations = list(itertools.product(*query_result.state_names.values()))
print("Top 3 Most Likely Outcomes:")
for i, index in enumerate(top_indices):
    most_likely_states = state_combinations[index]
    max_probability = query_result.values.flatten()[index]

    print(f"\nRank {i+1}:")
    for var, state in zip(query_result.state_names.keys(), most_likely_states):
        print(f"  {var}: {state}")
    print(f"  Probability: {max_probability:.4f}")