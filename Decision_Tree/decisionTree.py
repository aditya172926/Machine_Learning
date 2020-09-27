#Implementing the Decision Tree
training_dataset = [
    ['Green', 3, 'Mango'],
    ['Yellow', 3, 'Mango'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]   #[Feature, Feature, Label]

#Column labels
#These are used to printthe tree
header = ['color', 'diameter', 'label']

#A function of unique values
def unique_vals(rows, col):
    #Find the unique values in columns of a dataset
    return set([row[col] for row in rows])
#####Demo
#print(unique_vals(training_dataset, 0))
#print(unique_vals(training_dataset, 1))

def class_counts(rows):
    #Counts the number of each type of example in a dataset
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label]=0
        counts[label]+=1
    return counts
####Demo
#print(class_counts(training_dataset))

def is_numeric(value):
    #To test if the value is numeric
    return isinstance(value, int) or isinstance(value, float)
#print(is_numeric(7), is_numeric('Red'))

class Question:
    '''A question is used for partition a dataset.
This class just records a 'column number' (eg 0 for 'color') and a 'column value' (eg 'Green'). The 'match' method is
used to compare the feature value in an example to the feature value stored in the question.'''
    def __init__(self, column, value):
        self.column = column
        self.value = value
    def match(self, example):
        #Compare the feature value in this example to the feature value in this question
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        #This is just a helper method to print the question in the readable format
        condition = '=='
        if is_numeric(self.value):
            condition = '>='
        return ('Is %s %s %s?'%(header[self.column], condition, str(self.value)))

def partition(rows, question):
    '''Partitions a dataset.
        For each row in the dataset it checks if it matches the question. If so it adds to the 'true rows' otherwise
        it adds to the 'false rows'.
        '''
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

#true_rows, false_rows = partition(training_dataset, Question(0, 'Red'))
#print(true_rows, false_rows)
def gini(rows):
    #It will calculate the gini impurity for the list of rows
    
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl]/float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    '''Information gain
        The uncertainty of the starting node minus the weighted impurity of two child nodes'''
    p = float(len(left))/(len(left) + len(right))
    return current_uncertainty - p*gini(left) - (1-p)*gini(right)
'''Demo

Calculate the uncertainty of the training dataset
current_uncertainty = gini(training_dataset)
print(current_uncertainty)

How much information do we gain by partitioning on Green?

current_uncertainty = gini(training_dataset)
true_rows, false_rows = partition(training_dataset, Question(0, 'Green'))
print(info_gain(true_rows, false_rows, current_uncertainty))
print(current_uncertainty)
print(true_rows, false_rows)

print('\n')

Or how much information do we get partitioning by red

true_rowsr, false_rowsr = partition(training_dataset, Question(0, 'Red'))
print(info_gain(true_rowsr, false_rowsr, current_uncertainty))
print(true_rowsr, false_rowsr)'''

def find_best_split(rows):
    '''Find the best question to ask by iterating over every feature or value and calculating the information gain'''
    best_gain = 0   #Keep a track of best_gain of information gain
    best_question = None    #keep train of featrue or value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0])-1    #number of culomns

    for col in range(n_features):   #for each feature
        values = set([row[col] for row in rows])    #unique values in the column
        for val in values:  #for each value
            question = Question(col, val)

            #try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            #skip this split if it doesn't divide the dataset
            if len(true_rows)==0 or len(false_rows)==0:
                continue

            #calculate the information gain of this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question
'''Demo
best_gain, best_question = find_best_split(training_dataset)
print(best_gain, best_question)'''
            
        
















            
