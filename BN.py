import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# Step 1: Create the Bayesian Network
bn = BayesianNetwork([
    ('Flu', 'Fever'),
    ('Flu', 'Cough'),
    ('COVID-19', 'Fever'),
    ('COVID-19', 'Cough'),
    ('COVID-19', 'Fatigue'),
    ('Fever', 'Rest'),
    ('Fatigue', 'Rest'),
    ('Cough', 'Medication'),
    ('Rest', 'Recovery'),
    ('Medication', 'Recovery')
])

# Step 2: Define CPDs
cpd_Flu = TabularCPD(variable='Flu', variable_card=2, values=[[0.3], [0.7]])
cpd_COVID19 = TabularCPD(variable='COVID-19', variable_card=2, values=[[0.4], [0.6]])

cpd_Fever = TabularCPD(
    variable='Fever', variable_card=2,
    values=[
        [0.9, 0.6, 0.8, 0.5],  # P(Fever=0 | combinations of Flu and COVID-19)
        [0.1, 0.4, 0.2, 0.5]   # P(Fever=1 | combinations of Flu and COVID-19)
    ],
    evidence=['Flu', 'COVID-19'], evidence_card=[2, 2]
)

cpd_Cough = TabularCPD(
    variable='Cough', variable_card=2,
    values=[
        [0.8, 0.7, 0.6, 0.4],  # P(Cough=0 | combinations of Flu and COVID-19)
        [0.2, 0.3, 0.4, 0.6]   # P(Cough=1 | combinations of Flu and COVID-19)
    ],
    evidence=['Flu', 'COVID-19'], evidence_card=[2, 2]
)

cpd_Fatigue = TabularCPD(
    variable='Fatigue', variable_card=2,
    values=[
        [0.5, 0.8],  # P(Fatigue=0 | COVID-19)
        [0.5, 0.2]   # P(Fatigue=1 | COVID-19)
    ],
    evidence=['COVID-19'], evidence_card=[2]
)

cpd_Rest = TabularCPD(
    variable='Rest', variable_card=2,
    values=[
        [0.8, 0.6, 0.5, 0.3],  # P(Rest=0 | combinations of Fever and Fatigue)
        [0.2, 0.4, 0.5, 0.7]   # P(Rest=1 | combinations of Fever and Fatigue)
    ],
    evidence=['Fever', 'Fatigue'], evidence_card=[2, 2]
)

cpd_Medication = TabularCPD(
    variable='Medication', variable_card=2,
    values=[
        [0.7, 0.3],  # P(Medication=0 | Cough)
        [0.3, 0.7]   # P(Medication=1 | Cough)
    ],
    evidence=['Cough'], evidence_card=[2]
)

cpd_Recovery = TabularCPD(
    variable='Recovery', variable_card=2,
    values=[
        [0.9, 0.7, 0.6, 0.2],  # P(Recovery=0 | combinations of Rest and Medication)
        [0.1, 0.3, 0.4, 0.8]   # P(Recovery=1 | combinations of Rest and Medication)
    ],
    evidence=['Rest', 'Medication'], evidence_card=[2, 2]
)

# Add CPDs to the network
bn.add_cpds(cpd_Flu, cpd_COVID19, cpd_Fever, cpd_Cough, cpd_Fatigue, cpd_Rest, cpd_Medication, cpd_Recovery)

# Step 3: Validate the Bayesian Network
assert bn.check_model()

# Step 4: Arrange nodes for better visualization
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

# Visualize the Bayesian Network
nx_graph = nx.DiGraph(bn.edges)
plt.figure(figsize=(12, 8))
nx.draw(
    nx_graph, pos=positions, with_labels=True, node_size=3000,
    node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20
)
plt.title("Bayesian Network: Disease Diagnosis (Hierarchical View)")
plt.show()

# Step 5: Perform Inference
inference = VariableElimination(bn)

# Illnes diagnosis
print(inference.query(variables=['COVID-19'], evidence={'Fever': 1, 'Cough': 1}))
print(inference.query(variables=['Flu'], evidence={'Fever': 1, 'Cough': 1}))

# Prescribed treatment for recovery
print(inference.query(variables=['Rest', 'Medication'], evidence={'Fever': 1, 'COVID-19': 1, 'Recovery': 1}))
