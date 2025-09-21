#!/usr/bin/env python
# coding: utf-8

# In[2]:


def run_knn_analysis(filepath: str): 
    '''RUn the whole analysis
    '''
    X_train, X_test, y_train, y_test = dataset_preparation(filepath)
    run_knn_reg(X_train, X_test, y_train, y_test)

def run_gbr_analysis(filepath: str): 
    '''RUn the whole analysis
    '''
    X_train, X_test, y_train, y_test = dataset_preparation(filepath)
    run_mo_gbr(X_train, X_test, y_train, y_test)

def run_svr_mo_lr_analysis(filepath: str): 
    '''RUn the whole analysis
    '''
    X_train, X_test, y_train, y_test = dataset_preparation(filepath)
    run_mo_gbr_lr(X_train, X_test, y_train, y_test)

def run_rf_analysis(filepath: str): 
    '''RUn the whole analysis
    '''
    X_train, X_test, y_train, y_test = dataset_preparation(filepath)
    
    y_pred_test_mo, y_pred_test_rf = run__random_forests(X_train, y_train, X_test, y_test)

    evluate_results(y_pred_test_mo, y_pred_test_rf, y_test)


   
def smiles_to_ECFP(smiles: str) -> np.ndarray:
    '''Return the ecfp of smiles as a numpy array
    Precondition: smiles should be a string that corresponds to the SMILES of a molecule
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Generate Morgan fingerprint
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
    
    # Convert to numpy array
    ecfp_array = np.zeros((1024,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(ecfp, ecfp_array)
    return ecfp_array

def dataset_preparation(filepath: str):
    '''Return the X and y variables split in a 80/20 training/testing split'''
    df = pd.read_csv(filepath)
    
    # Drop rows with missing SMILES
    df = df.dropna(subset=["SMILES"])   

    # Find relevant columns
    smiles_collumn = df['SMILES']
    pdi_collumn = df['PDI']
    ee_collumn = df['EE%']
    pka_collumn = df['pKa']
    size_collumn = df['Size']

    # Convert SMILES → ECFP → 2D array
    X = np.vstack(smiles_collumn.apply(smiles_to_ECFP).to_numpy())

    # Build target matrix
    y = np.column_stack([pdi_collumn, ee_collumn, pka_collumn, size_collumn])
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Filter out rows with NaN targets
    mask_train = ~np.isnan(y_train).any(axis=1)
    X_train, y_train = X_train[mask_train], y_train[mask_train]

    mask_test = ~np.isnan(y_test).any(axis=1)
    X_test, y_test = X_test[mask_test], y_test[mask_test]

    return X_train, X_test, y_train, y_test

def run_knn_reg(X_train, X_test, y_train, y_test):
    '''Run KNN regression for k=1 and k=5, evaluate and plot results.'''
    
    for num in [1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12]: 
        reg = KNeighborsRegressor(n_neighbors=num)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred, multioutput="raw_values")
        mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")

        print(f"\nKNN with k={num}")
        targets = ["PDI", "EE%", "pKa", "Size"]
        for i, target in enumerate(targets):
            print(f"  {target}: R²={r2[i]:.3f}, MSE={mse[i]:.3f}")
           

        # Plot true vs predicted
        n_targets = y_test.shape[1]
        fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))

        if n_targets == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, label=f"k={num}")
            ax.plot([y_test[:, i].min(), y_test[:, i].max()],
                    [y_test[:, i].min(), y_test[:, i].max()], "k--", lw=2)
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title(f"{targets[i]}")
            ax.legend()

        plt.tight_layout()
        plt.show()

def run_mo_gbr(X_train, X_test, y_train, y_test):
    '''Run MultiOutputRegressor with Gradient Boosting Regressor.'''
    targets = ["PDI", "EE%", "pKa", "Size"]
    
    gbr = GradientBoostingRegressor(random_state=42)
    multi_gbr = MultiOutputRegressor(gbr)

    multi_gbr.fit(X_train, y_train)
    y_pred = multi_gbr.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")


    for i, target in enumerate(["PDI", "EE%", "pKa", "size"]):
        print(f"  {target}: R²={r2[i]:.3f}, MSE={mse[i]:.3f}")

    # Plot true vs predicted
    n_targets = y_test.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))

    if n_targets == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.6)
        ax.plot([y_test[:, i].min(), y_test[:, i].max()],
                [y_test[:, i].min(), y_test[:, i].max()], "k--", lw=2)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{targets[i]}")
        ax.legend()

    plt.tight_layout()
    plt.show()

def run_mo_svr_lr(X_train, X_test, y_train, y_test):
    '''Run MultiOutputRegressor with SVR, train-based linear calibration, metrics, and final plots.'''

    targets = ["PDI", "EE%", "pKa", "Size"]

    # Fit SVR wrapped in MultiOutputRegressor
    svr_model = MultiOutputRegressor(SVR())
    svr_model.fit(X_train, y_train)

    # SVR predictions
    y_pred_train = svr_model.predict(X_train)
    y_pred_test = svr_model.predict(X_test)

    n_targets = y_test.shape[1]
    y_pred_lr_all = np.zeros_like(y_pred_test)  # linear regression on test predictions
    y_pred_final = np.zeros_like(y_pred_test)   # final calibrated predictions

    # First linear regression: train, test calibration
    for i in range(n_targets):
        lr = LinearRegression()
        lr.fit(y_pred_train[:, i].reshape(-1, 1), y_train[:, i])
        y_pred_lr_all[:, i] = lr.predict(y_pred_test[:, i].reshape(-1, 1))

    # Compute metrics for linear regression calibration
    for i in range(n_targets):
        r2_lr = r2_score(y_test[:, i], y_pred_lr_all[:, i])
        rmse_lr = np.sqrt(mean_squared_error(y_test[:, i], y_pred_lr_all[:, i]))
        print(f"{targets[i]} - Linear regression calibration: R²={r2_lr:.3f}, RMSE={rmse_lr:.3f}")

    # Plot true vs calibrated predictions
    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))
    if n_targets == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.scatter(y_test[:, i], y_pred_lr_all[:, i], alpha=0.6, color='purple')
        ax.plot([y_test[:, i].min(), y_test[:, i].max()],
                [y_test[:, i].min(), y_test[:, i].max()], "k--", lw=2, label="y=x")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Calibrated Predictions")
        ax.set_title(f"{targets[i]} - Linear Regression Calibration")
        ax.legend()

    plt.tight_layout()
    plt.show()

    return y_pred_lr_all

def run__random_forests(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                        outer_cv=5, inner_cv=5, n_iter=20) -> tuple:
    '''Tune Random Forest + MultiOutput Random Forest with RandomizedSearchCV and GridSearchCV,
        return predictions on the held-out test set.'''
    
    #parameter space
    param_dist = {
        "n_estimators": [100, 200, 500, 800],
        "max_depth": [10, 20, 50, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }
    
    
    #cross validation splitter
    inner_cv_split = KFold(n_splits=inner_cv, shuffle=True, random_state=42)

    
    
    # NORMAL RANDOM FOREST
    rf = RandomForestRegressor(random_state=42)
    
    # Randomnized search 
    rand_search_rf = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=n_iter, cv=inner_cv_split, scoring="r2",
        n_jobs=-1, random_state=42
    )
    rand_search_rf.fit(X_train, y_train)
    
    # Refined grid search around best RF params
    best_params_rf = rand_search_rf.best_params_
    grid_params_rf = {
        "n_estimators": [max(50, best_params_rf["n_estimators"] - 50),
                         best_params_rf["n_estimators"],
                         best_params_rf["n_estimators"] + 50],
        "max_depth": [best_params_rf["max_depth"]] if best_params_rf["max_depth"] is None else [
            max(5, best_params_rf["max_depth"] - 10),
            best_params_rf["max_depth"],
            best_params_rf["max_depth"] + 10],
        "min_samples_split": [max(2, best_params_rf["min_samples_split"] - 1),
                              best_params_rf["min_samples_split"],
                              best_params_rf["min_samples_split"] + 1],
        "min_samples_leaf": [max(1, best_params_rf["min_samples_leaf"] - 1),
                             best_params_rf["min_samples_leaf"],
                             best_params_rf["min_samples_leaf"] + 1],
        "max_features": [best_params_rf["max_features"]]
    }
    
    grid_search_rf = GridSearchCV(rf, param_grid=grid_params_rf,
                                  cv=inner_cv_split, scoring="r2", n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    tuned_rf = grid_search_rf.best_estimator_
    
    # MULTI OUTPUT REGRESSOR
    mo_rf = MultiOutputRegressor(rf)

    # Randomnized search 
    rand_search_mo = RandomizedSearchCV(
        mo_rf, param_distributions={f"estimator__{k}": v for k, v in param_dist.items()},
        n_iter=n_iter, cv=inner_cv_split, scoring="r2",
        n_jobs=-1, random_state=42
    )
    rand_search_mo.fit(X_train, y_train)

     # Refined grid search around best MO RF params
    
    best_params_mo = rand_search_mo.best_params_
    grid_params_mo = {
        f"estimator__n_estimators": [max(50, best_params_mo["estimator__n_estimators"] - 50),
                                     best_params_mo["estimator__n_estimators"],
                                     best_params_mo["estimator__n_estimators"] + 50],
        f"estimator__max_depth": [best_params_mo["estimator__max_depth"]] if best_params_mo["estimator__max_depth"] is None else [
            max(5, best_params_mo["estimator__max_depth"] - 10),
            best_params_mo["estimator__max_depth"],
            best_params_mo["estimator__max_depth"] + 10],
        f"estimator__min_samples_split": [max(2, best_params_mo["estimator__min_samples_split"] - 1),
                                          best_params_mo["estimator__min_samples_split"],
                                          best_params_mo["estimator__min_samples_split"] + 1],
        f"estimator__min_samples_leaf": [max(1, best_params_mo["estimator__min_samples_leaf"] - 1),
                                         best_params_mo["estimator__min_samples_leaf"],
                                         best_params_mo["estimator__min_samples_leaf"] + 1],
        f"estimator__max_features": [best_params_mo["estimator__max_features"]]
    }
    
    grid_search_mo = GridSearchCV(mo_rf, param_grid=grid_params_mo,
                                  cv=inner_cv_split, scoring="r2", n_jobs=-1)
    grid_search_mo.fit(X_train, y_train)
    tuned_mo = grid_search_mo.best_estimator_
    
    # Evaluate test set
    y_pred_rf = tuned_rf.predict(X_test)
    y_pred_mo = tuned_mo.predict(X_test)
    
    # Log best params + Cross validation score
    print("\nBest RF Params:", grid_search_rf.best_params_)
    print("Best RF CV R²:", grid_search_rf.best_score_)
    
    print("\nBest MultiOutput RF Params:", grid_search_mo.best_params_)
    print("Best MultiOutput RF CV R²:", grid_search_mo.best_score_)
    
   # Decide which model is better
    if grid_search_rf.best_score_ >= grid_search_mo.best_score_:
        best_pred = y_pred_rf
        best_model_name = "Random Forest"
    else:
        best_pred = y_pred_mo
        best_model_name = "MultiOutput RF"

    # Plot only the best model
    # Plot both best models together in a single figure
    plot_best_models_together(y_test, y_pred_rf, y_pred_mo, target_names=["PDI", "EE%", "pKa", "Size"])
    
    # Return everything useful
    results = run__random_forests(X_train, y_train, X_test, y_test)

    y_pred_test_mo = results["y_pred_mo"]
    y_pred_test_rf = results["y_pred_rf"]
    
    return {
        "y_pred_rf": y_pred_rf,
        "y_pred_mo": y_pred_mo,
        "best_params_rf": grid_search_rf.best_params_,
        "best_params_mo": grid_search_mo.best_params_,
        "best_score_rf": grid_search_rf.best_score_,
        "best_score_mo": grid_search_mo.best_score_
    }



def plot_best_models_together(y_true: np.ndarray, y_pred_rf: np.ndarray, y_pred_mo: np.ndarray,
                              target_names=None):
    """
    Plot true vs predicted for each target in a single figure.
    RF points in red, MultiOutput RF points in blue.
    """
    n_targets = y_true.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 6))

    if n_targets == 1:
        axes = [axes]  # make iterable if only one target

    if target_names is None:
        target_names = [f"Target {i+1}" for i in range(n_targets)]

    for i, ax in enumerate(axes):
        # Scatter points for RF
        ax.scatter(y_true[:, i], y_pred_rf[:, i], color="red", alpha=0.6, label="RF")
        # Scatter points for MultiOutput RF
        ax.scatter(y_true[:, i], y_pred_mo[:, i], color="blue", alpha=0.6, label="MultiOutput RF")

        # Diagonal line
        min_val = min(y_true[:, i].min(), y_pred_rf[:, i].min(), y_pred_mo[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred_rf[:, i].max(), y_pred_mo[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{target_names[i]}: True vs Predicted")
        ax.legend()

    plt.tight_layout()
    plt.show()

