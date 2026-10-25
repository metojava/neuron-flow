# Java Machine Learning Framework

starter is at :  https://github.com/metojava/nfs-lib

A small Java 17 machine learning library built with Maven. The main module contains dense neural networks, activation and loss functions, optimizers, IDX dataset loaders, model serialization, and experimental autograd and transformer components. Example applications live under `projects/`; the older `chatgpt-worked/nfs` directory is a separate copy.

this is personal work by using AI : OpenCode and ChatGPT . project is working well 
but is in development process. not everyting can be perfect at this particular moment 
that is why if you/user/ decide to use in production or for any reason : 
take on your own risk. Author/Me/ not taking any kind of responsibility for anything .
God is main judge and I always remember about .
project was made with good intentions for anyones profit 
it is free to use for any reason and no liabilities considered in case of anything. 
air and sunlight is for everyone . same as AI for good reason


## Build and test

Install JDK 17 and Maven, then run from this directory:

```powershell
mvn clean test
```

The project uses the JDK incubating Vector API. Maven enables it for compilation and tests. When running a class directly with `java`, add `--add-modules jdk.incubator.vector`.

## Start here

The main API is `layers.NeuralNetwork`. See `src/main/java/examples/` for dataset runners and `newbook/index.html` for the detailed guide. The Fashion-MNIST demo downloads data on first run:

```powershell
mvn exec:exec
```

Model files can be saved as `.bin`, `.json`, or `.csv` with `NeuralNetwork.saveModel` and loaded with `NeuralNetwork.loadModel`. The caller supplies the same activation functions and training configuration when loading.

## Classical machine learning

The `ml` package offers a scikit-learn-style API for tabular data. `ml.DataFrame` and `ml.Vector` hold immutable numeric/categorical features and targets; `ml.Pipeline` chains transformers such as `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, and `MinMaxScaler`. Supervised models implement `Estimator`/`Predictor`, while `KMeans` implements `Clusterer`:

- `LinearRegression` and `LogisticRegression` use shuffled mini-batch SGD.
- `SVM` supports pluggable kernels (`LinearKernel`, `RbfKernel`) and multiclass one-vs-rest.
- `DecisionTree` supports Gini and entropy criteria; `RandomForest` adds bagging and random feature subsets.
- `KMeans` uses k-means++ seeding with restarts.

The end-to-end demo builds a mixed-type frame, fits a preprocessing pipeline on the training split, trains several classifiers, evaluates them, and then shows regression and clustering:

```powershell
mvn -q -DskipTests compile
java --add-modules jdk.incubator.vector -cp target/classes examples.ClassicalMlDemo
```

or dispatch it through the example runner with `examples.Main ml`. The full chapter is `newbook/16-classical-ml.html`, with a shorter current-API reference at `docs/html/classical-ml.html`. `newbook/21-api-reference.html` maps every package and records remaining documentation gaps.

## Data preparation and evaluation

`data.DatasetSplit.split(features, targets, testFraction, seed)` creates reproducible paired training and test sets. Fit `data.StandardScaler` on the training features only, then use that same scaler to transform the test features. `evaluation.RegressionMetrics` provides MAE, MSE, RMSE, and R² for single-output regression; `evaluation.ClassificationMetrics` covers classification. `layers.BatchNormalization` is a standalone batch utility with per-feature running statistics; its gamma and beta values are fixed parameters, and it is not wired into `NeuralNetwork` backpropagation.
