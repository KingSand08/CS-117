from DL_training_final_project import dl_builder
from k_NN_training_final_project import knn_builder
from format_data import format_data as data

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier ,StackingClassifier , AdaBoostClassifier
from sklearn.metrics import accuracy_score


train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = data()

# dl_model = dl_builder()
knn_model = knn_builder()

## create a voting classifier 
# model_list = [('dl',dl_model),('kn',knn_model)]
model_list = [('kn',knn_model)]

v = VotingClassifier(
    estimators = model_list , 
    n_jobs=-1
)

# train the voting classifier 
v.fit(train_inputs,train_targets)

# make predictions 
predicitons = v.predict(test_inputs)

# get model accuracy 
accuracy_score(test_targets,predicitons)
