# DM-HW-1-Q-2

This code has two main functionalities. The first function, `create_subset_files`, is designed to create smaller training files from a large original dataset. The second function is to `perform_cross_validation`, it then performs a k-fold cross-validation using ridge regression to select the best hyperparameter value for the regularization strength (`lambda`).

Here's a step-by-step explanation of each part of the code:

### create_subset_files Function
1. `original_file`: The CSV file containing the original dataset.
2. `instances_list`: A list containing different sizes of subsets to create from the original dataset.
3. The `pd.read_csv` method reads the original CSV file into a DataFrame.
4. The `for` loop iterates over each size specified in `instances_list`.
5. `data.head(instances)` selects the first 'instances' number of rows from the DataFrame.
6. `subset_file_name` creates a filename for the subset. It is formatted with the number of instances and fixed parts of the filename. The pattern assumes subsets will always have a "(1000)-100" naming scheme which might not be true for different datasets.
7. `subset.to_csv` writes the subset DataFrame to a CSV file without the index column.

### perform_cross_validation Function
1. `kf = KFold(...)`: It Initializes the KFold object for cross-validation with 10 splits and a fixed random state for reproducibility.
2. `lambda_values`: A list of potential regularization strengths (`alpha`) for the Ridge regression model.
3. The function iterates over each `lambda` value and performs cross-validation:
   - `mse_list` collects mean squared error (MSE) values for each fold.
   - The nested `for` loop uses the indices provided by `kf.split(X)` to split the data into training and validation sets (`X_train`, `X_val`, `y_train`, `y_val`).
   - A Ridge regression model is trained for each lambda value on the training set and then used to predict on the validation set.
   - The MSE for the validation predictions is computed and stored in `mse_list`.
   - The average MSE over all folds for the current lambda is calculated.
   - If the average MSE is smaller than the current best MSE, `best_lambda` and `best_mse` are updated.
4. The function returns `best_lambda` and `best_mse`, which represent the optimal lambda value and its corresponding MSE, respectively.

### Main Execution
- The `instances_list` is defined, with example subset sizes.
- `create_subset_files` function is called to create subset files.
- Placeholder numpy arrays `X_placeholder` and `y_placeholder` are created to simulate features and labels.
- The `perform_cross_validation` function is called with these placeholders and hypothetical `lambda_values`.
- Finally, it prints the best lambda value and its corresponding MSE found from cross-validation.

### Considerations and Usage
- The `placeholder` data should be replaced with actual feature matrices and label vectors from the prepared dataset.
- `lambda_values` should be a range that is sensible for the given problem or determined using domain knowledge.
- When using the `create_subset_files` function in practice, the naming pattern for the CSV files should adapt to the actual dataset, not just the predefined pattern.

This code is written in Python, utilizing libraries such as pandas for data handling, scikit-learn for machine learning algorithms and evaluation, and numpy for numerical operations. 

It is typical for machine learning applications where training on very large datasets and tuning regularization parameters are necessary tasks
