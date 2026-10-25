# A scikit-learn-style API for tabular data

*The `layers` package is built for images and dense neural networks. The `ml` package is built for rows and columns: mixed numeric/categorical features, missing values, train/test splits, linear models, trees, kernels, and clustering. It borrows scikit-learn's vocabulary — *estimator*, *transformer*, *pipeline* — without hiding the math.*

## 16.1 Why a second API?

Tabular problems have different requirements from image problems. A spreadsheet column can be numeric or categorical, can be missing, and can have a wildly different scale from its neighbour. A neural network can absorb raw numbers, but a linear model or an SVM usually cannot. The `ml` package therefore makes preprocessing a first-class object and validates schemas instead of silently reshaping arrays.

| Package | Data type | Best for | Entry point |
| --- | --- | --- | --- |
| `layers` | `float[]` / `float[][]` | Images, dense and convolutional networks | `NeuralNetwork` |
| `autograd`, `transformers` | `float` tensors | Sequence models and custom graphs | `Tensor`, `Ops` |
| `ml` | `double` columns | Tabular regression, classification, clustering | `DataFrame` + `Pipeline` |
| `data` | `float[][]` | CSV/Parquet/IDX loading, CV folds, scaling | loaders and splitters |

> **Two numeric worlds.** The classical API uses `double` for numerically sensitive linear algebra; the neural-network stack uses `float` for SIMD throughput. Convert at the boundary with `DataFrame.toDoubleMatrix()` and your own float loop; the two packages intentionally do not depend on each other.

## 16.2 DataFrame and Vector

`ml.DataFrame` is an immutable, column-oriented table. Every column is either `NUMERIC` (a `double[]`, with `Double.NaN` as the missing marker) or `CATEGORICAL` (a `String[]`, with `null` as the missing marker). Accessors return defensive copies, so a frame can be shared across pipeline stages without aliasing bugs.

```
DataFrame raw = DataFrame.builder(4)
        .addNumeric("age",    new double[]{34, 51, Double.NaN, 28})
        .addNumeric("income", new double[]{62_000, 91_000, 47_000, Double.NaN})
        .addCategorical("city", new String[]{"Paris", "Berlin", null, "Paris"})
        .addCategorical("plan", new String[]{"premium", "basic", "premium", null})
        .build();

int rows = raw.rows();          // 4
int columns = raw.columns();    // 4
String[] names = raw.names();   // {"age", "income", "city", "plan"}
ColumnType[] types = raw.types();

double age = raw.getDouble(0, 0);
String city = raw.getString(0, 2);
double[] ages = raw.numericColumn(0);
int cityIndex = raw.columnIndex("city");
```

| Method | Purpose |
| --- | --- |
| `DataFrame.ofNumeric(names, rows)` | Build a numeric-only frame from a row-major matrix. |
| `DataFrame.ofCategorical(names, rows)` | Build a categorical-only frame. |
| `DataFrame.builder(rows)` | Build a mixed frame one column at a time. |
| `select(int...)`, `takeRows(int[])` | Column and row selection, preserving types. |
| `appendNumericColumn(name, values)` | Add an engineered feature. |
| `toDoubleMatrix()` | Row-major matrix; fails if any column is categorical. |
| `allNumeric()` | True when the frame can feed a model directly. |

`ml.Vector` is the matching immutable target container. Labels are `double` values, so `0/1`, `-1/+1` and arbitrary class ids all work. Useful methods are `get`, `take`, `toArray`, `count`, `uniqueSorted`, `mean` and `requireFinite`.

## 16.3 Preprocessing transformers

Every transformer implements the same three-method contract:

```
public interface Transformer {
    Transformer fit(DataFrame X);          // learn parameters
    DataFrame transform(DataFrame X);      // apply them
    default DataFrame fitTransform(DataFrame X) { return fit(X).transform(X); }
    boolean isFitted();
}
```

Fitting must happen on the training split only. The fitted object is then reused unchanged for validation, test and production data. Calling `transform` before `fit` throws `IllegalStateException`; changing the column names or types after fitting throws `IllegalArgumentException`.

### SimpleImputer

Numeric columns use `Double.NaN`; categorical columns use `null`. The numeric strategy is `MEAN` (default), `MEDIAN` or `MOST_FREQUENT`. Categorical columns always use the most frequent category because mean and median are undefined for strings. A column with no observed values is filled with `0` or `"__missing__"`.

```
Transformer imputer = new SimpleImputer(SimpleImputer.Strategy.MEDIAN);
DataFrame filled = imputer.fitTransform(raw);
double[] statistics = ((SimpleImputer) imputer).numericStatistics();
```

### OneHotEncoder

Categorical columns become indicator columns named `column[category]`; numeric columns pass through. Categories are learned in sorted order, which makes output names deterministic. By default the first category is dropped to avoid perfect collinearity in linear models. At transform time an unseen category becomes all zeros, or raises an error if you choose `UnknownStrategy.ERROR`.

```
OneHotEncoder encoder = new OneHotEncoder(true, OneHotEncoder.UnknownStrategy.IGNORE);
DataFrame encoded = encoder.fitTransform(filled);
List<String> produced = encoder.outputNames("city");
// [city[Berlin], city[Paris]]
```

> **One category only.** With `dropFirst=true`, a categorical column that contains a single category produces no columns. If every categorical column has one category, `transform` throws because the result would be empty.

### StandardScaler and MinMaxScaler

`StandardScaler` computes `z = (x - mean) / std`, using the population standard deviation. A constant numeric column gets scale `1`, so it is centred but not divided by zero. `MinMaxScaler` maps values into a configurable range (default `[0, 1]`); a constant column maps to the lower bound. Both pass categorical columns through untouched, so they can sit in a mixed pipeline, and both reject non-finite numeric values: impute first.

```
Pipeline pipeline = Pipeline.of(
        new SimpleImputer(SimpleImputer.Strategy.MEAN),
        new OneHotEncoder(),
        new StandardScaler());

DataFrame train = pipeline.fitTransform(split.train());
DataFrame test  = pipeline.transform(split.test());
```

`Pipeline.fit` fits each stage on the *output* of the previous one, then stores the fitted stages. This ordering is important: scaling before imputation would compute statistics that include missing markers, and encoding before imputation would create a `null` category.

## 16.4 Train/test split

`ml.DatasetSplit.split(X, y, testFraction, seed)` shuffles row indices with a seeded `Random`, then returns a `Split` record with `train()`, `yTrain()`, `test()`, `yTest()`, `trainSize()` and `testSize()`. The fraction must lie strictly between 0 and 1; at least one row is kept on each side.

```
DatasetSplit.Split split = DatasetSplit.split(raw, labels, 0.25, 42L);
```

## 16.5 The estimators

Supervised models implement `Estimator<M>` and `Predictor`: `fit` returns the fitted model, `predict` returns one value per row, and `predictProbabilities` returns per-class probabilities (or a single column for regression). Fitted getters throw `IllegalStateException` before training. Every estimator also validates that prediction columns have the same names and order as training.

| Model | Task | Training | Key hyperparameters |
| --- | --- | --- | --- |
| `LinearRegression` | Regression | Mini-batch SGD | `learningRate`, `maxEpochs`, `l2`, `tolerance`, `batchSize`, `seed`, `fitIntercept` |
| `LogisticRegression` | Classification | Mini-batch SGD, one-vs-rest | same set |
| `DecisionTree` | Classification | CART splits | `criterion`, `maxDepth`, `minSamplesSplit`, `minSamplesLeaf`, `maxFeatures`, `seed` |
| `RandomForest` | Classification | Bagging + feature subsets | `numberOfTrees`, tree parameters, `maxFeatures`, `seed` |
| `SVM` | Classification | SMO, one-vs-rest | `c`, `kernel`, `tolerance`, `maxPasses` |
| `KMeans` | Clustering | Lloyd + k-means++ | `k`, `maxIterations`, `tolerance`, `restarts`, `seed` |

### LinearRegression

Fits `y = w · x + b` by minimising mean squared error plus an optional L2 penalty. The gradient for a mini-batch is

∂L/∂w = (1/B) Σ (ŷ - y) x + λ w ∂L/∂b = (1/B) Σ (ŷ - y)

Rows are reshuffled every epoch with the supplied seed. Training stops early when the epoch loss changes by at most `tolerance`. The default configuration is `learningRate=0.05`, `maxEpochs=2000`, `l2=0`, `tolerance=1e-10`, `batchSize=32`, `seed=42`, with an intercept. Gradient descent is scale sensitive, so standardise features first. `score` returns R²; `coefficients()`, `intercept()` and `epochsRun()` expose the fit.

### LogisticRegression

Binary problems fit one sigmoid model for the larger label; multiclass problems fit one sigmoid per class and normalise the sigmoid outputs at prediction time. The loss is binary log-loss with optional L2. Defaults are `learningRate=0.1`, `maxEpochs=1000`, `tolerance=1e-8`, `batchSize=32`, `seed=42`. `classes()` returns ascending labels, and probability columns follow that order. Multiclass probabilities are normalised but not calibrated.

### DecisionTree

A CART-style binary tree. At each node it sorts each candidate feature, tries every midpoint between adjacent distinct values, and keeps the split with the largest weighted impurity decrease. Gini impurity is

Gini = 1 - Σ pk2 Entropy = -Σ pk log2 pk

Leaves store the empirical class distribution, so `predictProbabilities` returns smoothed counts rather than hard labels. Defaults are Gini, `maxDepth=10`, `minSamplesSplit=2`, `minSamplesLeaf=1`, all features, seed 42. `featureImportances()` reports the normalised impurity decrease attributed to each feature. A single tree cannot extrapolate beyond the training range.

### RandomForest

Fits many decision trees on bootstrap samples; each split considers a random subset of features. The default feature subset is `round(sqrt(numberOfFeatures))`. Predictions average the trees' class distributions, which gives smoother probabilities than one tree. The default forest has 100 trees, Gini, depth 8, `minSamplesSplit=2`, `minSamplesLeaf=1`, and seed 42. `featureImportances()` averages the per-tree importances.

### SVM

A kernel support-vector machine trained with sequential minimal optimisation. Binary problems use one decision function; multiclass problems use one-vs-rest. Kernels implement `ml.kernel.Kernel`:

```
Kernel linear = Kernel.linear();                 // a · b
Kernel rbf    = Kernel.rbf(0.25);                // exp(-gamma ||a - b||²)
Kernel smart  = RbfKernel.withDefaultGamma(6);   // gamma = 1 / features

SVM svm = new SVM(10.0, linear, 1e-4, 100).fit(train, split.yTrain());
```

`c` trades margin width against training errors; larger values fit the training data more closely. The kernel matrix is precomputed, costing `O(n²)` memory, so this implementation targets small and medium datasets. `supportVectorCounts()` reports how many support vectors each one-vs-rest model kept. Probabilities are sigmoid-transformed decision values, useful for ranking but not calibrated.

### KMeans

Unsupervised clustering with k-means++ seeding and multiple restarts. Each restart runs Lloyd's algorithm until the largest centre shift is at most `tolerance`; the restart with the lowest inertia wins. Empty clusters are repaired by moving them onto the worst-fit point. Defaults are `maxIterations=300`, `tolerance=1e-4`, `restarts=10`, seed 42. `fitPredict` fits and returns zero-based labels in one call. Standardise features first when their units differ.

## 16.6 End-to-end demo

`examples.ClassicalMlDemo` generates a mixed-type purchase dataset with missing values, fits the preprocessing pipeline on the training split, trains four classifiers, then runs a regression and a clustering example. The essential classification code is:

```
DataFrame raw = DataFrame.builder(rows)
        .addNumeric("age", age)
        .addNumeric("income", income)
        .addNumeric("score", score)
        .addCategorical("city", city)
        .addCategorical("plan", plan)
        .build();
Vector labels = Vector.of(purchased);
DatasetSplit.Split split = DatasetSplit.split(raw, labels, 0.25, 42L);

Pipeline pipeline = Pipeline.of(
        new SimpleImputer(SimpleImputer.Strategy.MEAN),
        new OneHotEncoder(),
        new StandardScaler());
DataFrame train = pipeline.fitTransform(split.train());
DataFrame test  = pipeline.transform(split.test());

LogisticRegression logistic = new LogisticRegression(0.2, 3_000, 0.0, 1e-10, true)
        .fit(train, split.yTrain());
SVM svm = new SVM(10.0, Kernel.linear(), 1e-4, 100)
        .fit(train, split.yTrain());
DecisionTree tree = new DecisionTree(DecisionTree.Criterion.GINI, 6)
        .fit(train, split.yTrain());
RandomForest forest = new RandomForest(80, DecisionTree.Criterion.GINI, 8, 2, 1,
        DecisionTree.ALL_FEATURES, 42L).fit(train, split.yTrain());

double accuracy = Metrics.accuracy(split.yTest(), logistic.predict(test));
double[][] probabilities = logistic.predictProbabilities(test);
```

Run it from the project root:

```
mvn -q -DskipTests compile
java --add-modules jdk.incubator.vector -cp target/classes examples.ClassicalMlDemo
```

or through the dispatcher: `examples.Main ml`. Representative output:

```
-- Classification: predicting purchases --
Raw columns: 5, engineered columns: 6, train/test: 300/100
LogisticRegression       accuracy = 0.950
SVM (linear kernel)      accuracy = 0.930
DecisionTree             accuracy = 0.850
RandomForest             accuracy = 0.940

-- Regression: predicting house prices --
Train/test: 135/45, R2 = 0.891, RMSE = 8998

-- Clustering: k-means on three blobs --
Inertia = 14.165, iterations = 3, cluster sizes = [60, 60, 60]
```

In the classification task the raw frame has five columns; one-hot encoding turns them into six numeric columns. The forest's top importances are `income (0.57)`, `age (0.24)`, `score (0.14)` and `plan[premium] (0.02)`, which matches the generative process.

## 16.7 Metrics

| Metric | Signature | Interpretation |
| --- | --- | --- |
| Accuracy | `Metrics.accuracy(yTrue, yPred)` | Fraction of exact matches; classification. |
| Mean squared error | `Metrics.meanSquaredError(yTrue, yPred)` | Average squared residual; lower is better. |
| R² | `Metrics.r2Score(yTrue, yPred)` | 1 is perfect, 0 predicts the mean, negative is worse than the mean. |

accuracy = correct / N MSE = (1/N) Σ (y - ŷ)2 R² = 1 - Σ(y - ŷ)2 / Σ(y - ȳ)2

For richer evaluation of the neural-network stack, see `evaluation.ClassificationMetrics` (confusion matrix, precision/recall/F1), `evaluation.RegressionMetrics` (MAE, MSE, RMSE, R²), `evaluation.AucMetrics`, and the bootstrap helpers.

## 16.8 Pitfalls and design rules

- **Fit on train, transform test.** Never call `fitTransform` on the test split; that leaks statistics.
- **Models need numeric frames.** Encode categorical columns and impute missing values first. `ModelUtils` rejects non-finite features with a message naming the column.
- **Column names and order are checked.** A model trained on `[age, income]` will refuse a frame whose first column is `income`.
- **Scaling is not optional for linear models and SVMs.** Income in thousands can dominate age in years; standardise first.
- **Trees ignore scale** but still reject missing values. Impute, then fit.
- **SVM memory grows quadratically.** A 50,000-row kernel matrix has 2.5 billion entries; subsample or use a linear solver for large data.
- **Probabilities are not calibrated.** Logistic and SVM probabilities rank examples well but should not be read as exact frequencies without a calibration step.
- **Seeds make runs reproducible.** Splits, shuffles, forest bootstrap samples, k-means++ seeds and tree feature subsets all take a seed.
- **Class labels are doubles.** `1.0` and `1` are the same label; `0.999999` is a different one. Use exact values.

## 16.9 Quick reference

| Class | Important methods |
| --- | --- |
| `DataFrame` | `builder`, `ofNumeric`, `ofCategorical`, `rows`, `columns`, `names`, `types`, `numericColumn`, `categoricalColumn`, `select`, `takeRows`, `toDoubleMatrix` |
| `Vector` | `of`, `copyOf`, `zeros`, `get`, `take`, `toArray`, `uniqueSorted`, `mean` |
| `Pipeline` | `of`, `then`, `fit`, `transform`, `fitTransform`, `transformers` |
| `DatasetSplit` | `split` → `Split.train`, `yTrain`, `test`, `yTest` |
| `LinearRegression` | `fit`, `predict`, `score`, `coefficients`, `intercept`, `epochsRun` |
| `LogisticRegression` | `fit`, `predict`, `predictProbabilities`, `score`, `classes`, `coefficients`, `intercepts` |
| `DecisionTree` | `fit`, `predict`, `predictProbabilities`, `score`, `classes`, `featureImportances` |
| `RandomForest` | `fit`, `predict`, `predictProbabilities`, `score`, `featureImportances` |
| `SVM` | `fit`, `predict`, `predictProbabilities`, `score`, `supportVectorCounts`, `biases` |
| `KMeans` | `fit`, `predict`, `fitPredict`, `inertia`, `clusterCenters`, `iterationsRun` |

## 16.10 Exercises

1. Add an `AgeBands` feature that buckets age into `"young"`, `"middle"` and `"senior"`, then compare logistic accuracy with and without it.
2. Swap the linear SVM for `RbfKernel.withDefaultGamma(train.columns())` and tune `c` over `{0.1, 1, 10, 100}` using the test split.
3. Implement a `Transformer` that standardises only selected numeric columns and leaves the rest alone.
4. Run `DatasetSplit.split` with ten different seeds and report the mean and standard deviation of random-forest accuracy.
5. Replace `Metrics.accuracy` with a macro-F1 score computed from the confusion matrix and explain why the ranking of models can change.

---

[← 15 · Practice](15-practice.md) · [17 · Autograd →](17-autograd.md)
