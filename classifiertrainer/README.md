# classifiertrainer
This is a small module that trains and evaluates classifiers on the Iris dataset. Currently only KNN classifiers are supported.

# Install instructions
Install via git
```
git clone git@github.com:pmocherla/ai-library.git ai-library
cd ai-library/classifiertrainer
python setup.py install
```
To test things are running run from your python interpreter
```
from example import example_training_func
example_training_func()
```
or from command line run
```
python example.py 
```

# Example usage
```
from classifiertrainer.iris_data_handler import IrisDataHandler
from classifiertrainer.classifier_trainer import KNNTrainer

iris_data = IrisDataHandler(seed=8)
X_train, X_test, y_train, y_test = iris_data.split_data()

params = {'n_neighbors' : 5}
knn = KNNTrainer(params)
knn.train(X_train, y_train)
y_pred, results_dict = knn.evaluate(X_test, y_test)
```