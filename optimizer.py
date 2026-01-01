import pandas as pd
import pandas_ta as ta
import os
import pickle
import signal
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from main import main as run_backtest_for_params


EXIT_FLAG = False

def save_trials(trials, filename):
    """Safely save trials object to disk"""
    with open(filename, 'wb') as f:
        pickle.dump(trials, f)
    print(f"Saved checkpoint with {len(trials.trials)} evaluations")

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    global EXIT_FLAG
    EXIT_FLAG = True
    print("\nInterrupt received - saving current state before exit")




# This dictionary defines the "search space" for the optimizer.
# It specifies the range and distribution for each parameter.
hyperparameter_space = {
    # State Classification Parameters
    'atr_period': hp.quniform('atr_period', 10, 20, 1),
    'price_state_threshold': hp.uniform('price_state_threshold', 0.5, 1.25),
    'volume_high_multiplier': hp.uniform('volume_high_multiplier', 1.5, 3.0),
    'volume_low_multiplier': hp.uniform('volume_low_multiplier', 0.6, 1.0),

    # Pattern Recognition Parameters
    'sequence_length': hp.quniform('sequence_length', 5, 8, 1),
    'min_signal_probability': hp.uniform('min_signal_probability', 0.60, 0.85),

    # Risk Management Parameters
    'sl_multiplier': hp.uniform('sl_multiplier', 1.5, 3.5),
    'tp_multiplier': hp.uniform('tp_multiplier', 1.5, 4.0),

    # Trend Filter Parameter
    'trend_period_sma': hp.quniform('trend_period_sma', 150, 250, 10),
}


# --- Phase 3: Create the "Objective Function" ---

def objective(params):
    """
    The objective function for hyperopt to minimize.
    Now optimized for Sharpe and Sortino ratios.
    """
    # hyperopt provides float values from quniform, so we must cast them to int
    params['atr_period'] = int(params['atr_period'])
    params['sequence_length'] = int(params['sequence_length'])
    params['trend_period_sma'] = int(params['trend_period_sma'])

    print("\n--- Testing Parameters ---")
    print(params)

    try:
        # Load data
        dataframe = pd.read_feather('SOL_USDT-1h.feather')

        # The main function now handles all data preparation
        final_metrics = run_backtest_for_params(
            df=dataframe.copy(),
            start_year=2021,
            end_year=2025,
            save_dir="hyperopt_temp_results",
            params=params
        )

        sharpe = final_metrics.get('sharpe_ratio', 0)
        sortino = final_metrics.get('sortino_ratio', 0)

        # Combine ratios (you can adjust weights if you prefer one over the other)
        combined_score = sharpe

        # Since hyperopt minimizes, we return negative of our desired metric
        loss = -combined_score  # Negative because we want to maximize

        print(
            f"Result: Sharpe = {sharpe:.4f}, Sortino = {sortino:.4f}, Combined = {combined_score:.4f}, Loss = {loss:.4f}")
        return {
            'loss': loss,
            'status': STATUS_OK,
            'metrics': final_metrics,
            'params': params,
            'sharpe': sharpe,
            'sortino': sortino
        }

    except Exception as e:
        print(f"An error occurred during backtest with params {params}: {e}")
        return {'loss': 9999.0, 'status': 'fail', 'params': params}


if __name__ == "__main__":
    trials_file = "eth_hyperopt_trials.pkl"
    max_evals = 1000

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load existing trials or create new
    if os.path.exists(trials_file):
        with open(trials_file, "rb") as f:
            trials = pickle.load(f)
        print(f"Resuming from {len(trials.trials)} completed evaluations")
    else:
        trials = Trials()
        print("Starting new optimization")

    # Manual optimization loop
    start_index = len(trials.trials)
    for i in range(start_index, max_evals):
        if EXIT_FLAG:
            print("Early termination requested")
            break

        # Run one evaluation at a time
        best = fmin(
            fn=objective,
            space=hyperparameter_space,
            algo=tpe.suggest,
            max_evals=len(trials.trials) + 1,  # Run just one more evaluation
            trials=trials,
            show_progressbar=False
        )

        # Save after every 5 evaluations
        if i % 5 == 0:
            save_trials(trials, trials_file)

        # Print progress
        print(f"\nCompleted {i + 1}/{max_evals} ({((i + 1) / max_evals) * 100:.1f}%)")
        print(f"Current best loss: {trials.best_trial['result']['loss']:.4f}")

    # Final save
    save_trials(trials, trials_file)

    print("\n\n--- Hyperopt Complete ---")
    print("Best parameters found:")
    print(best)

    # The 'best' dictionary only contains the raw values.
    # We can get the full details of the best run from the trials object.
    best_trial = trials.best_trial
    best_params = best_trial['result']['params']
    best_metrics = best_trial['result']['metrics']

    print("\n--- Best Trial Details ---")
    print(f"Loss (Negative Profit Factor): {best_trial['result']['loss']:.4f}")
    print("Optimal Parameters:")
    print(best_params)
    print("\nPerformance Metrics:")
    print(best_metrics)

    # Save all trial results to a CSV for later analysis
    results_df = pd.DataFrame([t['result'] for t in trials.trials])
    # Expand the dictionaries into separate columns
    params_df = pd.json_normalize(results_df['params'])
    metrics_df = pd.json_normalize(results_df['metrics'])
    final_df = pd.concat([results_df.drop(columns=['params', 'metrics']), params_df, metrics_df], axis=1)
    final_df.sort_values(by='loss', ascending=True, inplace=True)
    final_df.to_csv("full_hyperopt_results.csv", index=False)
    print("\nFull optimization results saved to 'full_hyperopt_results.csv'")


def analyze_partial_results(trials):
    if len(trials.trials) == 0:
        return

    # Get all completed runs
    results = [trial['result'] for trial in trials.trials
               if trial['result'].get('status') == STATUS_OK]

    if not results:
        return

    # Create dataframe
    df = pd.DataFrame({
        'loss': [r['loss'] for r in results],
        'params': [r['params'] for r in results]
    })

    # Add parameter columns
    param_df = pd.json_normalize(df['params'])
    result_df = pd.concat([df['loss'], param_df], axis=1)

    # Show best 5 configurations
    print("\nTop configurations so far:")
    print(result_df.sort_values('loss').head(5))

    # Basic statistics
    print(f"\nCompleted runs: {len(results)}")
    print(f"Best loss: {result_df['loss'].min()}")
    print(f"Recent loss: {results[-1]['loss']}")
