from optuna_dl_search import run_study as optuna_dl_test
from optuna_knn_search import run_study as optuna_knn_test
from optuna_ensemble_voting import run_study as optuna_voting_test
from optuna_ensemble_dl_bagging import run_study as optuna_dl_bagging_test
from optuna_ensemble_knn_bagging import run_study as optuna_knn_bagging_test

# optuna_dl_test(1000)

# optuna_knn_test(1000)

# optuna_voting_test(1000)
optuna_dl_bagging_test(1000)
# optuna_knn_bagging_test(1000)
