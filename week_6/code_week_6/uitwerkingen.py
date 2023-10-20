import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# OPGAVE 1a
def plot_image(img, label):
    # Deze methode krijgt een matrix mee (in img) en een label dat correspondeert met het 
    # plaatje dat in de matrix is weergegeven. Zorg ervoor dat dit grafisch wordt weergegeven.
    # Maak gebruik van plt.cm.binary voor de cmap-parameter van plt.imgshow.

    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(label)
    plt.savefig(f"plotted_image.png", dpi=300)


# OPGAVE 1b
def scale_data(X):
    # Deze methode krijgt een matrix mee waarin getallen zijn opgeslagen van 0..m, en hij 
    # moet dezelfde matrix retourneren met waarden van 0..1. Deze methode moet werken voor 
    # alle maximale waarde die in de matrix voorkomt.
    # Deel alle elementen in de matrix 'element wise' door de grootste waarde in deze matrix.

    X_max = np.max(X) # np.max is een alias voor numpy.amax
    X_scaled = X / X_max
    return X_scaled

# OPGAVE 1c
def build_model():
    # Deze methode maakt het keras-model dat we gebruiken voor de classificatie van de mnist
    # dataset. Je hoeft deze niet abstract te maken, dus je kunt er van uitgaan dat de input
    # layer van dit netwerk alleen geschikt is voor de plaatjes in de opgave (wat is de 
    # dimensionaliteit hiervan?).
    # Maak een model met een input-laag, een volledig verbonden verborgen laag en een softmax
    # output-laag. Compileer het netwerk vervolgens met de gegevens die in opgave gegeven zijn
    # en retourneer het resultaat.

    # Het staat je natuurlijk vrij om met andere settings en architecturen te experimenteren.

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)), # zet de 28x28 matrix om in een 784x1 vector
        keras.layers.Dense(128, activation='relu'), # eerste hidden layer met 128 neuronen, maakt gebruik van ReLU. https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        keras.layers.Dense(10, activation='softmax') # output layer met 10 neuronen, maakt gebruik van softmax. https://en.wikipedia.org/wiki/Softmax_function
    ])

    model.compile(
        optimizer='adam', # https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/#:~:text=Adam%20optimizer%2C%20short%20for%20Adaptive%20Moment%20Estimation%20optimizer%2C%20is%20an%20optimization%20algorithm%20commonly%20used%20in%20deep%20learning.%20It%20is%20an%20extension%20of%20the%20stochastic%20gradient%20descent%20(SGD)%20algorithm%20and%20is%20designed%20to%20update%20the%20weights%20of%20a%20neural%20network%20during%20training.
        loss='sparse_categorical_crossentropy', # https://rmoklesur.medium.com/what-you-need-to-know-about-sparse-categorical-cross-entropy-9f07497e3a6f#:~:text=Sparse%20categorical%20cross%2Dentropy%20is%20an%20extension%20of%20the%20categorical,a%20one%2Dhot%20encoded%20vector.
        metrics=['accuracy'] # https://en.wikipedia.org/wiki/Accuracy_and_precision
    )

    return model


# OPGAVE 2a
def conf_matrix(labels, pred):
    # Retourneer de econfusion matrix op basis van de gegeven voorspelling (pred) en de actuele
    # waarden (labels). Check de documentatie van tf.math.confusion_matrix:
    # https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
    
    conf = tf.math.confusion_matrix(labels, pred)
    return conf
    

# OPGAVE 2b
def conf_els(conf, labels): 
    # Deze methode krijgt een confusion matrix mee (conf) en een set van labels. Als het goed is, is 
    # de dimensionaliteit van de matrix gelijk aan len(labels) × len(labels) (waarom?). Bereken de 
    # waarden van de TP, FP, FN en TN conform de berekening in de opgave. Maak vervolgens gebruik van
    # de methodes zip() en list() om een list van len(labels) te retourneren, waarbij elke tupel 
    # als volgt is gedefinieerd:

    #     (categorie:string, tp:int, fp:int, fn:int, tn:int)
 
    # Check de documentatie van numpy diagonal om de eerste waarde te bepalen.
    # https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html
 
    # TP zijn de waarden op de diagonaal van de matrix
    tp = np.diagonal(conf)
    # FP zijn de waarden in de kolommen van de matrix, min de TP
    fp = np.sum(conf, axis=0) - tp
    # FN zijn de waarden in de rijen van de matrix, min de TP
    fn = np.sum(conf, axis=1) - tp
    # TN zijn alle waarden in de matrix, min de TP, FP en FN
    tn = np.sum(conf) - tp - fp - fn

    return list(zip(labels, tp, fp, fn, tn))

# OPGAVE 2c
def conf_data(metrics):
    # Deze methode krijgt de lijst mee die je in de vorige opgave hebt gemaakt (dus met lengte len(labels))
    # Maak gebruik van een list-comprehension om de totale tp, fp, fn, en tn te berekenen en 
    # bepaal vervolgens de metrieken die in de opgave genoemd zijn. Retourneer deze waarden in de
    # vorm van een dictionary (de scaffold hiervan is gegeven).

    # VERVANG ONDERSTAANDE REGELS MET JE EIGEN CODE
    
    sensitivity = metrics[0][1] / (metrics[0][1] + metrics[0][3])
    precision = metrics[0][1] / (metrics[0][1] + metrics[0][2])
    specificity = metrics[0][4] / (metrics[0][4] + metrics[0][2])
    fall_out = metrics[0][2] / (metrics[0][2] + metrics[0][4])

    # BEREKEN HIERONDER DE JUISTE METRIEKEN EN RETOURNEER DIE 
    # ALS EEN DICTIONARY

    rv = {'tpr':sensitivity, 'ppv':precision, 'tnr':specificity, 'fpr':fall_out }
    return rv
    # opmerking: is het best-practice om een variable te returnen in plaats van, in dit geval, gewoon `return {'tpr':0, 'ppv':0, 'tnr':0, 'fpr':0 }`?
