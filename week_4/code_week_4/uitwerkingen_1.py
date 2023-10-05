import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import binary_classification as bc

learning_rate = 0.04
iterations = 100

# Exercise 1
X, y = datasets.load_iris(return_X_y=True)

y[y != 0] -= 1
y = y.reshape(-1, 1)

t = np.ones((X.shape[1], 1))

costs = []

for _ in range(iterations):
    # Calculate the cost for the current theta
    costs.append(bc.cost(X, y, t))

    # Calculate the gradient
    gradient = bc.gradient(X, y, t)

    # Update theta with the gradient
    t = t - learning_rate * gradient
    h = bc.sigmoid(np.dot(X, t))

print(f'Begin cost: {costs[0]}')
print(f'Final cost: {costs[-1]}')
print(f'Minimum cost: {min(costs)}')

# Plot the costs
plt.figure()
plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.savefig('exercise_1_1.png', dpi=300)
print('Figure saved to exercise_1.png')

"""
Wat opvalt in de grafiek is dat het een mooie zogeheten "elleboog" is. Dit betekent dat de kosten in het begin snel afnemen, maar dat dit na een tijdje minder snel gaat. Dit is een goed teken, want dit betekent dat de kosten afnemen en dat de theta-waarden dus steeds beter worden.
Wanneer de delta tussen de kosten steeds kleiner wordt, betekent dit dat de theta-waarden hoogst waarschijnlijk niet veel beter gaan worden. Om deze reden zou in dit geval de aantal iteraties eventueel verminderd kunnen worden naar bijvoorbeeld 100, gezien hier de kosten al op 0.42 liggen. Ook voorkom je op deze manier dat er overfitting plaatsvindt.
Ook kan de alpha (learing-rate) hoger worden gezet, het feit dat het nogsteeds 100 iteraties duurt voordat de kosten op 0.42 liggen betekent dat de alpha nog hoger kan worden gezet. Tenzij er een hoge precisie nodig is, echter is de dataset hier te klein (150 waarnemningen) voor.
"""