from surprise.model_selection import cross_validate


def evaluate_models(data, models):
    results = {}
    for name, model in models.items():
        cv_results = cross_validate(model, data, measures=['RMSE'], cv=3, verbose=False)
        results[name] = cv_results['test_rmse'].mean()
    return results
