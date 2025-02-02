import networkx as nx               # For creating and visualizing graphs.
import matplotlib.pyplot as plt       # For plotting the network graph.
from pgmpy.models import BayesianNetwork  # To create and work with Bayesian networks.
from pgmpy.inference import VariableElimination  # For performing probabilistic inference.
from pgmpy.factors.discrete import TabularCPD  # To define Conditional Probability Distributions (CPDs).
import numpy as np                    # For numerical operations and array handling.
import math                           # For mathematical functions (like logarithms).

# =============================================================================
# STEP 1: IMPORT LIBRARIES
# =============================================================================
# Simple Terms:
#   We import libraries that help us create graphs, plot them, define a Bayesian network,
#   perform inference, and handle mathematical operations.
#
# Advanced Details:
#   - networkx is used to construct and manipulate the graph structure of the network.
#   - matplotlib.pyplot helps us visualize the network layout.
#   - pgmpy is a specialized library for probabilistic graphical models, providing tools for
#     building Bayesian networks, defining CPDs, and running inference algorithms like Variable
#     Elimination.
#   - numpy and math are used for numerical computations, such as computing entropy.

# =============================================================================
# STEP 2: CREATE THE BAYESIAN NETWORK
# =============================================================================
# Simple Terms:
#   We create a network where nodes represent events (diseases, symptoms, etc.) and edges
#   represent influences between them.
#
# Advanced Details:
#   A Bayesian network is a Directed Acyclic Graph (DAG) where each node is a random variable,
#   and the edges encode conditional dependencies between variables. The absence of an edge
#   implies a conditional independence assumption.
#
# Our network structure:
#   - 'Flu' and 'COVID-19' are diseases.
#   - 'Fever', 'Cough', and 'Fatigue' are symptoms influenced by these diseases.
#   - 'Rest' and 'Medication' are treatment-related nodes influenced by symptoms.
#   - 'Recovery' is an outcome influenced by treatment and symptom-related nodes.
bn = BayesianNetwork([
    ('Flu', 'Fever'),        # Simple: Flu can cause Fever.
    ('Flu', 'Cough'),        # Simple: Flu can cause Cough.
    ('COVID-19', 'Fever'),   # Simple: COVID-19 can cause Fever.
    ('COVID-19', 'Cough'),   # Simple: COVID-19 can cause Cough.
    ('COVID-19', 'Fatigue'), # Simple: COVID-19 can cause Fatigue.
    ('Fever', 'Rest'),       # Simple: Fever might lead to the need for Rest.
    ('Fatigue', 'Rest'),     # Simple: Fatigue might also lead to taking Rest.
    ('Cough', 'Medication'), # Simple: Cough might lead to Medication.
    ('Rest', 'Recovery'),    # Simple: Rest can influence Recovery.
    ('Medication', 'Recovery')  # Simple: Medication can influence Recovery.
])

# =============================================================================
# STEP 3: DEFINE THE CPDs (CONDITIONAL PROBABILITY DISTRIBUTIONS)
# =============================================================================
# Simple Terms:
#   CPDs tell us how likely each node is to be in a certain state, optionally given the state of its parent nodes.
#
# Advanced Details:
#   For each node X with parents Y, the CPD represents the conditional probabilities P(X|Y). For binary nodes,
#   we use two states (often 0 and 1). The values in the CPD are arranged so that each column corresponds
#   to a unique combination of parent states.
#
# For our network, we assume:
#   - For disease nodes ('Flu', 'COVID-19'): a CPD is defined without any evidence (they are root nodes).
#   - For symptom nodes ('Fever', 'Cough', 'Fatigue'): their CPDs depend on the relevant diseases.
#   - For treatment and outcome nodes ('Rest', 'Medication', 'Recovery'): their CPDs depend on symptoms/treatment.
#
# Define CPD for 'Flu'.
#   - In simple terms: there's a 30% chance of one state and a 70% chance of the other.
#   - Advanced: The values [[0.3], [0.7]] imply P(Flu=0)=0.3 and P(Flu=1)=0.7.
cpd_Flu = TabularCPD(variable='Flu', variable_card=2, values=[[0.3], [0.7]])

# Define CPD for 'COVID-19'.
cpd_COVID19 = TabularCPD(variable='COVID-19', variable_card=2, values=[[0.4], [0.6]])

# Define CPD for 'Fever', which depends on both 'Flu' and 'COVID-19'.
#   - There are 4 combinations of parent states (Flu and COVID-19 each have 2 states, so 2x2=4).
#   - Simple: For each combination, we specify the chance of Fever being absent (state 0) or present (state 1).
#   - Advanced: Each column in the values array corresponds to a particular configuration of (Flu, COVID-19).
cpd_Fever = TabularCPD(
    variable='Fever', variable_card=2,
    values=[
        [0.9, 0.6, 0.8, 0.5],  # P(Fever=0 | Flu, COVID-19) for each combination.
        [0.1, 0.4, 0.2, 0.5]   # P(Fever=1 | Flu, COVID-19) for each combination.
    ],
    evidence=['Flu', 'COVID-19'], evidence_card=[2, 2]
)

# Define CPD for 'Cough', also influenced by 'Flu' and 'COVID-19'.
cpd_Cough = TabularCPD(
    variable='Cough', variable_card=2,
    values=[
        [0.8, 0.7, 0.6, 0.4],  # P(Cough=0 | Flu, COVID-19)
        [0.2, 0.3, 0.4, 0.6]   # P(Cough=1 | Flu, COVID-19)
    ],
    evidence=['Flu', 'COVID-19'], evidence_card=[2, 2]
)

# Define CPD for 'Fatigue', which depends solely on 'COVID-19'.
cpd_Fatigue = TabularCPD(
    variable='Fatigue', variable_card=2,
    values=[
        [0.5, 0.8],  # P(Fatigue=0 | COVID-19)
        [0.5, 0.2]   # P(Fatigue=1 | COVID-19)
    ],
    evidence=['COVID-19'], evidence_card=[2]
)

# Define CPD for 'Rest', which depends on 'Fever' and 'Fatigue'.
cpd_Rest = TabularCPD(
    variable='Rest', variable_card=2,
    values=[
        [0.8, 0.6, 0.5, 0.3],  # P(Rest=0 | Fever, Fatigue)
        [0.2, 0.4, 0.5, 0.7]   # P(Rest=1 | Fever, Fatigue)
    ],
    evidence=['Fever', 'Fatigue'], evidence_card=[2, 2]
)

# Define CPD for 'Medication', which depends only on 'Cough'.
cpd_Medication = TabularCPD(
    variable='Medication', variable_card=2,
    values=[
        [0.7, 0.3],  # P(Medication=0 | Cough)
        [0.3, 0.7]   # P(Medication=1 | Cough)
    ],
    evidence=['Cough'], evidence_card=[2]
)

# Define CPD for 'Recovery', which depends on 'Rest' and 'Medication'.
cpd_Recovery = TabularCPD(
    variable='Recovery', variable_card=2,
    values=[
        [0.9, 0.7, 0.6, 0.2],  # P(Recovery=0 | Rest, Medication)
        [0.1, 0.3, 0.4, 0.8]   # P(Recovery=1 | Rest, Medication)
    ],
    evidence=['Rest', 'Medication'], evidence_card=[2, 2]
)

# =============================================================================
# STEP 4: ADD THE CPDs TO THE NETWORK AND VALIDATE THE MODEL
# =============================================================================
# Simple Terms:
#   We add all the CPDs to the network and then check if everything is set up correctly.
#
# Advanced Details:
#   The check_model() function verifies that the network is well-defined:
#     - Every node has an associated CPD.
#     - The CPDs match the network structure (e.g., the dimensions and parent configurations are correct).
bn.add_cpds(cpd_Flu, cpd_COVID19, cpd_Fever, cpd_Cough, cpd_Fatigue, cpd_Rest, cpd_Medication, cpd_Recovery)
assert bn.check_model(), "The model is invalid!"

# =============================================================================
# STEP 5: VISUALIZE THE BAYESIAN NETWORK
# =============================================================================
# Simple Terms:
#   We create a visual diagram of the network to see how the nodes are connected.
#
# Advanced Details:
#   Using NetworkX, we construct a directed graph (DiGraph) based on the edges defined in the Bayesian network.
#   The 'positions' dictionary specifies the (x, y) coordinates for each node for clearer visualization.
positions = {
    'Flu': (0, 4),
    'COVID-19': (2, 4),
    'Fever': (1, 3),
    'Cough': (3, 3),
    'Fatigue': (2, 3),
    'Rest': (1.5, 2),
    'Medication': (3, 2),
    'Recovery': (2, 1),
}
nx_graph = nx.DiGraph(bn.edges)   # Create a directed graph from the Bayesian network edges.
plt.figure(figsize=(12, 8))       # Set the figure size for better clarity.
nx.draw(
    nx_graph, pos=positions, with_labels=True, node_size=3000,
    node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20
)
plt.title("Bayesian Network: Disease Diagnosis (Hierarchical View)")
plt.show()  # Display the network graph.

# =============================================================================
# STEP 6: PERFORM INFERENCE USING THE BAYESIAN NETWORK
# =============================================================================
# Simple Terms:
#   We now ask questions about the network. For example, if we know that a patient has Fever and Cough,
#   how likely is it that they have COVID-19 or Flu?
#
# Advanced Details:
#   Variable Elimination is an exact inference algorithm that sums out (marginalizes) variables not of interest.
#   This produces the posterior probability distribution for the target variable given some evidence.
inference = VariableElimination(bn)

# Example Query 1:
#   Given Fever=1 and Cough=1, compute the posterior probability for 'COVID-19'.
print("Posterior for COVID-19 given Fever=1 and Cough=1:")
print(inference.query(variables=['COVID-19'], evidence={'Fever': 1, 'Cough': 1}))

# Example Query 2:
#   Given the same evidence, compute the posterior probability for 'Flu'.
print("\nPosterior for Flu given Fever=1 and Cough=1:")
print(inference.query(variables=['Flu'], evidence={'Fever': 1, 'Cough': 1}))

# Example Query 3:
#   Compute the joint probability for 'Rest' and 'Medication' given Fever=1, COVID-19=1, and Recovery=1.
print("\nPosterior for Rest and Medication given Fever=1, COVID-19=1 and Recovery=1:")
print(inference.query(variables=['Rest', 'Medication'], evidence={'Fever': 1, 'COVID-19': 1, 'Recovery': 1}))

# =============================================================================
# STEP 7: VALUE-OF-INFORMATION (VOI) ANALYSIS
# =============================================================================
# Simple Terms:
#   VOI analysis helps decide which extra piece of information (which additional node) would most help us
#   be more certain about our target variable—in this case, 'COVID-19'.
#
# Advanced Details:
#   We measure uncertainty using entropy. Entropy quantifies how spread out (or uncertain) a probability distribution is.
#   By computing the expected entropy of 'COVID-19' after observing an additional candidate node, we can
#   determine which node would reduce our uncertainty the most. The candidate with the highest information gain,
#   defined as the reduction in entropy, is the most valuable.
#
# Define a helper function to compute entropy.
def entropy(prob_dist):
    """
    Compute the entropy (in bits) of a probability distribution.
    
    Simple Terms:
    - Entropy tells us how uncertain a probability distribution is.
    - A distribution like [0.5, 0.5] (equal chances) has high uncertainty, while [0.99, 0.01] has low uncertainty.
    
    Advanced Details:
    - The entropy is calculated as: -sum(p * log2(p)) for each probability p.
    - It originates from information theory and quantifies the expected amount of "information" (or surprise)
      when sampling from the distribution.
    
    :param prob_dist: A list or numpy array of probabilities (which should sum to 1).
    :return: The entropy (in bits).
    """
    prob_dist = np.array(prob_dist)
    return -np.sum([p * math.log2(p) for p in prob_dist if p > 0])

# Define a function to compute the expected entropy of the target node after observing a candidate node.
def expected_entropy_for_node(candidate_node, current_evidence, target_node):
    """
    Calculate the expected entropy of the target node after we add the observation of a candidate node.
    
    Simple Terms:
    - For each possible outcome of the candidate node, we:
      1. Assume that outcome happens.
      2. Compute the updated probability (posterior) of the target node.
      3. Calculate the entropy of this new distribution.
    - We then average these entropies, weighting each by the chance that the outcome occurs.
    
    Advanced Details:
    - Let p(state) be the probability that the candidate node is in a given state.
    - For each state, we compute P(target_node | current evidence + candidate=state).
    - The expected entropy is: Σ [p(state) * Entropy(P(target_node | candidate=state))].
    
    :param candidate_node: The node we are considering to observe.
    :param current_evidence: Dictionary of the evidence we already have.
    :param target_node: The node whose uncertainty we want to reduce (e.g., 'COVID-19').
    :return: The expected entropy of the target node after observing candidate_node.
    """
    # Query the marginal distribution of the candidate node given current evidence.
    marginal = inference.query(variables=[candidate_node], evidence=current_evidence)
    marginal_probs = marginal.values  # Probabilities for each state of the candidate node.
    
    exp_entropy = 0.0  # Initialize the expected entropy accumulator.
    
    # Iterate through each possible state of the candidate node.
    for state in range(marginal_probs.shape[0]):
        p_state = marginal_probs[state]  # The probability that candidate_node is in this state.
        
        # Create a new evidence set that includes the candidate node's observed state.
        new_evidence = current_evidence.copy()
        new_evidence[candidate_node] = state
        
        # Query the posterior distribution for the target node with the new evidence.
        posterior = inference.query(variables=[target_node], evidence=new_evidence)
        target_probs = posterior.values
        
        # Compute the entropy for this new posterior distribution.
        h = entropy(target_probs)
        
        # Multiply by the probability of the candidate node's state and add to expected entropy.
        exp_entropy += p_state * h
    return exp_entropy

# ----------------------------
# VOI ANALYSIS SETTINGS
# ----------------------------
# Set the target node for which we want to reduce uncertainty.
target_node = 'COVID-19'
# Define the evidence we already have. Here, we know that Fever=1 and Cough=1.
current_evidence = {'Fever': 1, 'Cough': 1}

# Compute the current posterior for the target node given the current evidence.
posterior_target = inference.query(variables=[target_node], evidence=current_evidence)
target_probs = posterior_target.values

# Calculate the baseline entropy for the target node using the current evidence.
baseline_entropy = entropy(target_probs)
print("\nCurrent posterior for '{}': {}".format(target_node, target_probs))
print("Baseline entropy for '{}': {:.4f} bits".format(target_node, baseline_entropy))

# ----------------------------
# IDENTIFY CANDIDATE NODES FOR FURTHER OBSERVATION
# ----------------------------
# We consider all nodes that are not already observed and are not our target node.
all_nodes = set(bn.nodes())
observed_nodes = set(current_evidence.keys())
candidate_nodes = all_nodes - observed_nodes - {target_node}
print("\nCandidate nodes for additional observation:", candidate_nodes)

# ----------------------------
# COMPUTE EXPECTED ENTROPY AND INFORMATION GAIN FOR EACH CANDIDATE
# ----------------------------
# For each candidate node:
#   1. Compute the expected entropy of the target node after observing it.
#   2. Compute the information gain (the reduction in entropy).
info_gain = {}
for candidate in candidate_nodes:
    # Calculate the expected entropy if we were to observe this candidate node.
    exp_ent = expected_entropy_for_node(candidate, current_evidence, target_node)
    # The information gain is the difference between the baseline entropy and the expected entropy.
    gain = baseline_entropy - exp_ent
    info_gain[candidate] = gain
    print("Candidate: {:12s} | Expected entropy: {:.4f} bits | Information Gain: {:.4f} bits"
          .format(candidate, exp_ent, gain))

# ----------------------------
# SELECT THE BEST CANDIDATE
# ----------------------------
# The candidate node that yields the highest information gain is the most valuable additional observation.
if info_gain:
    best_candidate = max(info_gain, key=info_gain.get)
    print("\nRecommended additional observation: '{}'".format(best_candidate))
else:
    print("\nNo candidate nodes available for additional observation.")
