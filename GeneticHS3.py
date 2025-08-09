import numpy as np
import os
import tensorflow as tf
import random
import warnings
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
warnings.filterwarnings("ignore")

class GeneticHyperparametersSearch:
    def __init__(self, train_data, train_label, test_data, test_label, validation_data, validation_label):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.validation_data = validation_data
        self.validation_label = validation_label
        self.population = []
        self.phase = 0
        self.generation=0
        self.autoModel=None
        self.auto_parameter=None
        self.max_size=None
    
    def initializzation(self, n_pop, max_layers, max_dim):
        models=[[random.randint(10,max_dim) for j in range(0,random.randint(2,max_layers))] for x in range(n_pop)]
        
        for chromosome in models:
            fitness= encoder(chromosome,self.train_data, self.train_label, self.test_data, self.test_label,
                self.validation_data, self.validation_label )
            #print(fitness)
            self.population.append((chromosome, fitness))

        self.population=sorted(self.population, key=lambda x: x[1], reverse=True)
    
    def start_evolution(self, n_childs, mutation_rate, mutation_magnitude, elitism, survival_rate=75):
        for i in range(3):
            self.generation=i
            self.mating(n_childs, mutation_rate, mutation_magnitude, elitism, survival_rate)
            avg_acc = sum(acc for _, acc in self.population) / len(self.population)
            print(f'Generation {i+1} avg_accuracy: {avg_acc} max_accuracy: {self.population[0][1]} population {len(self.population)}')
        return self.population[0]

    def mating(self, n_childs, mutation_rate, mutation_magnitude, elitism, survival_rate=75):
        self.MetaModel(self.population)

        new_chromosomes = []
        for m in range(len(self.population)):
            for _ in range(n_childs):
                if elitism:
                    mating_choice = random.randint(0, round(len(self.population) * 0.2))
                else:
                    mating_choice = random.randint(0, len(self.population) - 1)

                while mating_choice == m:
                    if elitism:
                        mating_choice = random.randint(0, round(len(self.population) * 0.2))
                    else:
                        mating_choice = random.randint(0, len(self.population) - 1)

                chrom1, _ = self.population[m]
                chrom2, _ = self.population[mating_choice]

                if self.phase == 0:
                    hybrid_chrom = self.merge_phase(chrom1, chrom2)
                else:
                    hybrid_chrom = self.split_phase(chrom1, chrom2, mutation_rate, mutation_magnitude)

                if self.generation >=0:
                    to_evaluate=[x/self.auto_parameter for x in hybrid_chrom]
                    while len(to_evaluate) < self.max_size:
                        to_evaluate.append(0)

                    prediction=self.autoModel.predict(np.array([to_evaluate[:self.max_size]]), verbose=0)

                    avg=(round(len(self.population) * (10/100)))
                    if avg==0:
                        _, top_10=self.population[0]
                    else:
                        top_10 = sum(acc for _, acc in self.population[:round(len(self.population)* (10/100) )]) / avg
                    if prediction[0,0]*100 < top_10*100:

                        pass
                    else:
                        #print('accepted', hybrid_chrom, prediction, top_10)
                        new_chromosomes.append(hybrid_chrom)

                else:
                    new_chromosomes.append(hybrid_chrom)
        print(f'{len(new_chromosomes)} accepeted out of {len(self.population)* n_childs}')
        self.phase = (self.phase + 1) % 2

        
        for chromosome in new_chromosomes:
            fitness= encoder(chromosome,self.train_data, self.train_label, self.test_data, self.test_label,
                self.validation_data, self.validation_label )
            self.population.append((chromosome, fitness))


        self.natural_selection(survival_rate)
    
    def natural_selection(self, survival_rate):
        self.population = sorted(self.population, key=lambda x: x[1], reverse=True)
        survived_index = round(len(self.population) * (survival_rate / 100))
        self.population = self.population[:survived_index]

        
    def MetaModel(self,data):
        y_label=[]
        X=[]
       
        for d, y in data:
            y_label.append(y)
            X.append(d)


        raveled=max([elem for row in X for elem in row ])
        max_size=max([len(row) for row in X])
        self.auto_parameter=raveled
        self.max_size=max_size
        edit_data=[]
        
        for row in X:
            p=[]
            for elem in row:
                p.append(elem/raveled)
            while len(p)<max_size:
                p.append(0)
            edit_data.append(p)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(1))

        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        edit_data=np.array(edit_data)
        y_label=np.array(y_label)

        X_train, X_test, y_train, y_test = train_test_split(edit_data, y_label, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 
        model.fit(X_train, y_train, epochs=25, validation_data=(X_val, y_val),verbose=0)
        self.autoModel=model 
        



    def merge_phase(self, chrom1, chrom2):
        if len(chrom1) < len(chrom2):
            shorter, longer = chrom1, chrom2
        else:
            shorter, longer = chrom2, chrom1
            
        common_length = len(shorter)
        new_param = [round((shorter[x] + longer[x]) / 2) for x in range(common_length)]
        new_param.extend(longer[common_length:])
        return new_param
    
    def split_phase(self, m1, m2, mutation_rate, mutation_magnitude):
        i=0 
        if len(m1) > len(m2):
            cut_index = max(1, round(len(m1) * 0.25))
            hybrid = m2[:cut_index]
           
            while len(hybrid) < 3:
                hybrid.append(m1[i])
                i+=1
        else:
            cut_index = max(1, round(len(m2) * 0.25))
            hybrid = m1[:cut_index]
            while len(hybrid) < 3:
                hybrid.append(m2[i])
                i+=1

        if random.randint(0, 100) < mutation_rate:
            mutation_vector = [random.randint(round(-hybrid[x] * (mutation_magnitude / 100)),
                                              round(hybrid[x] * (mutation_magnitude / 100))) for x in range(len(hybrid))]
            
            hybrid = [mutation_vector[x] + hybrid[x] for x in range(len(hybrid))]
        return hybrid


def encoder(chromosome, train_data, train_labels, test_data, test_labels, validation_data, validation_label):
    n_classes = len(np.unique(train_labels))
    model = tf.keras.models.Sequential()
    activation_functions = ['relu', 'sigmoid', 'tanh', 'elu', 'selu', 'softplus', 'swish']
    for units in chromosome:
        model.add(tf.keras.layers.Dense(units, activation=random.choice(activation_functions)))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=25, validation_data=(validation_data, validation_label), verbose=0)
    fitness = model.evaluate(test_data, test_labels, verbose=0)[1]
    tf.keras.backend.clear_session()
    #print(fitness)
    return fitness

def decoder(chromosome):
    return chromosome




if __name__ == '__main__':
    X, labels=make_classification(n_samples=1000, n_features=10, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)
    X_train, X_validation, y_train, y_validation=train_test_split(X_train, y_train, test_size=0.20, random_state=42)



    A = GeneticHyperparametersSearch(X_train, y_train, X_test, y_test, X_validation, y_validation)
    #n_pop, max_layers, max_dim
    A.initializzation(40, 4, 128)
    best_chrom, best_fit = A.start_evolution(5, 10, 20, False, 50)

    final_model= encoder(best_chrom, X_train, y_train, X_test, y_test, X_validation, y_validation)
    print(final_model, best_chrom)
