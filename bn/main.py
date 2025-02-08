from typing import Dict, Set, Tuple

from pyvis.network import Network
import networkx as nx
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel



def network_train(evidence_binary: pd.DataFrame) -> Tuple[BayesianModel, VariableElimination]:
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



def network_visualisation(model: BayesianModel) -> None:
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
    """Compute the Shannon entropy (in bits) of a probability distribution.

    The entropy is defined as: H(X) = -sum(p * log2(p)) for all p > 0.
    See: https://en.wikipedia.org/wiki/Entropy_(information_theory)

    Args:
        prob_dist (np.ndarray): An array of probabilities (should sum to 1).

    Returns:
        float: The entropy in bits.
    """
    prob_dist = np.array(prob_dist)
    return -np.sum([p * math.log2(p) for p in prob_dist if p > 0])


def expected_entropy_for_node(candidate_node: str,
                              current_evidence: dict,
                              target_node: str,
                              inference: VariableElimination) -> float:
    """Calculate the expected entropy of the target node after observing a candidate node.

    For each possible state of the candidate node, this function:
      1. Computes the probability of that state given the current evidence.
      2. Updates the evidence with the candidate node's state.
      3. Calculates the entropy of the target node's posterior distribution.
      4. Averages these entropies weighted by the probability of each state.

    Args:
        candidate_node (str): The variable being considered for additional observation.
        current_evidence (dict): Existing evidence as a dictionary (e.g., {'Fever': 1, 'Cough': 1}).
        target_node (str): The target variable whose uncertainty is to be reduced.
        inference (VariableElimination): The inference engine for the Bayesian network.

    Returns:
        float: The expected entropy of the target node after observing the candidate node.
    """
    # Get the marginal probability distribution for the candidate node.
    marginal = inference.query(variables=[candidate_node], evidence=current_evidence)
    marginal_probs = marginal.values

    exp_entropy = 0.0
    # Evaluate the effect of each possible state of the candidate node.
    for state in range(marginal_probs.shape[0]):
        p_state = marginal_probs[state]
        new_evidence = current_evidence.copy()
        new_evidence[candidate_node] = state

        # Compute the posterior for the target node with the updated evidence.
        print(new_evidence)
        print(target_node)
        posterior = inference.query(variables=[target_node], evidence=new_evidence)
        target_probs = posterior.values

        # Calculate the entropy of the updated posterior and weight by the state's probability.
        h = entropy(target_probs)
        exp_entropy += p_state * h

    return exp_entropy



def perform_value_of_information(inference: VariableElimination,
                                 model: BayesianModel,
                                 target_node: str,
                                 current_evidence: dict) -> Dict[str, float]:
    """Compute the information gain for candidate nodes as additional observations.

    This function calculates the current posterior and baseline entropy for the target node.
    Then, for each candidate node (nodes not in the current evidence and not the target), it computes
    the expected entropy of the target node if that candidate were observed.

    Args:
        inference (VariableElimination): The inference engine for the Bayesian network.
        model (BayesianModel): The Bayesian network model.
        target_node (str): The target variable for which we wish to reduce uncertainty.
        current_evidence (dict): The currently observed evidence.

    Returns:
        Dict[str, float]: A dictionary mapping candidate nodes to their information gain (in bits).
    """
    # Compute the current posterior for the target node.
    posterior_target = inference.query(variables=[target_node], evidence=current_evidence)
    target_probs = posterior_target.values
    baseline_entropy = entropy(target_probs)
    print("\nCurrent posterior for '{}': {}".format(target_node, target_probs))
    print("Baseline entropy for '{}': {:.4f} bits".format(target_node, baseline_entropy))

    # Identify candidate nodes: all nodes except those already observed and the target node.
    all_nodes: Set[str] = set(model.nodes())
    observed_nodes: Set[str] = set(current_evidence.keys())
    candidate_nodes: Set[str] = all_nodes - observed_nodes - {target_node}
    print("\nCandidate nodes for additional observation:", candidate_nodes)

    info_gain: Dict[str, float] = {}
    # Compute expected entropy and information gain for each candidate node.
    for candidate in candidate_nodes:
        exp_ent = expected_entropy_for_node(candidate, current_evidence, target_node, inference)
        gain = baseline_entropy - exp_ent
        info_gain[candidate] = gain
        print("Candidate: {:12s} | Expected entropy: {:.4f} bits | Information Gain: {:.4f} bits"
              .format(candidate, exp_ent, gain))
    return info_gain



evidence_binary = pd.read_csv('data/df_train.csv')

# Train the Bayesian network and create an inference engine.
model, inference = network_train(evidence_binary)

# Visualize the learned Bayesian network.
network_visualisation(model)

# Define the target node and current evidence for inference.
target_node = "diagnoses_category_gi"
current_evidence = {"lab_category_platelets": 1, "lab_category_bun": 1}

# Compute the information gain for candidate nodes.
info_gain = perform_value_of_information(inference, model, target_node, current_evidence)

if info_gain:
    best_candidate = max(info_gain, key=info_gain.get)
    print("\nRecommended additional observation: '{}'".format(best_candidate))
else:
    print("\nNo candidate nodes available for additional observation.")