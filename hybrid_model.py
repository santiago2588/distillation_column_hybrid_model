
import joblib
import pandas as pd
from first_principles_model import first_principles_distillation_model

def hybrid_distillation_model(operating_conditions, alpha=0.5):
    # Load the trained Random Forest model
    rf_model = joblib.load(
        '/home/ubuntu/random_forest_model.joblib'
    )

    # Load feature names used during training
    with open('/home/ubuntu/rf_model_features.txt', 'r') as f:
        rf_features = [line.strip() for line in f]

    # Prepare input for Random Forest model
    # Ensure the order of features matches the training data
    rf_input = pd.DataFrame([operating_conditions], columns=rf_features)
    ml_predictions = rf_model.predict(rf_input)[0]
    ml_mole_fraction_tx = ml_predictions[0]
    ml_mole_fraction_hx = ml_predictions[1]

    # Extract relevant parameters for the first principles model
    # These mappings need to be accurate based on your dataset columns
    feed_mole_fraction = operating_conditions[
        'Feed Mole Fraction'
    ]  # Assuming this column exists
    reflux_ratio = operating_conditions[
        'Reflux Ratio'
    ]  # Assuming this column exists
    # Placeholder values for num_stages, condenser_pressure, reboiler_pressure, feed_temperature
    # These would ideally come from operating_conditions or be fixed design parameters
    num_stages = 20  # Example value
    condenser_pressure = operating_conditions[
        'Condenser Pressure'
    ]  # Example value
    reboiler_pressure = operating_conditions[
        'Bottom Tower Pressure'
    ]  # Example value
    feed_temperature = operating_conditions[
        'Feed Tray Temperature'
    ]  # Example value

    (fp_mole_fraction_tx, fp_mole_fraction_hx) = first_principles_distillation_model(
        feed_mole_fraction,
        reflux_ratio,
        num_stages,
        condenser_pressure,
        reboiler_pressure,
        feed_temperature,
    )

    # Combine predictions (simple weighted average)
    final_mole_fraction_tx = (
        alpha * fp_mole_fraction_tx + (1 - alpha) * ml_mole_fraction_tx
    )
    final_mole_fraction_hx = (
        alpha * fp_mole_fraction_hx + (1 - alpha) * ml_mole_fraction_hx
    )

    return final_mole_fraction_tx, final_mole_fraction_hx


# Example usage (for testing purposes)
# if __name__ == "__main__":
#     # Create a dummy operating_conditions dictionary matching the features used in RF model
#     # This needs to be populated with realistic values from your dataset description
#     dummy_operating_conditions = {
#         'Liquid Percentage in Condenser': 50.0,
#         'Condenser Pressure': 100.0,
#         'Liquid Percentage in Reboiler': 50.0,
#         'Mass Flow Rate in Feed Flow': 1000.0,
#         'Mass Flow Rate in Top Outlet Stream': 500.0,
#         'Net Mass Flow in main tower': 500.0,
#         'Mole Fraction HX at reboiler': 0.05,
#         'HX Mole Fraction in Top Outler Stream': 0.02,
#         'Feed Mole Fraction': 0.5,
#         'Feed Tray Temperature': 350.0,
#         'Main Tower Pressure': 100.0,
#         'Bottom Tower Pressure': 102.0,
#         'Top Tower Pressure': 98.0,
#         'Reflux Ratio': 2.5
#     }

#     predicted_tx, predicted_hx = hybrid_distillation_model(dummy_operating_conditions)
#     print(f"Hybrid Predicted MoleFractionTX: {predicted_tx}")
#     print(f"Hybrid Predicted MoleFractionHX: {predicted_hx}")




if __name__ == "__main__":
    # Create a dummy operating_conditions dictionary matching the features used in RF model
    # This needs to be populated with realistic values from your dataset description
    dummy_operating_conditions = {
        'Liquid Percentage in Condenser': 50.0,
        'Condenser Pressure': 100.0,
        'Liquid Percentage in Reboiler': 50.0,
        'Mass Flow Rate in Feed Flow': 1000.0,
        'Mass Flow Rate in Top Outlet Stream': 500.0,
        'Net Mass Flow in main tower': 500.0,
        'Mole Fraction HX at reboiler': 0.05,
        'HX Mole Fraction in Top Outler Stream': 0.02,
        'Feed Mole Fraction': 0.5,
        'Feed Tray Temperature': 350.0,
        'Main Tower Pressure': 100.0,
        'Bottom Tower Pressure': 102.0,
        'Top Tower Pressure': 98.0,
        'Reflux Ratio': 2.5
    }

    predicted_tx, predicted_hx = hybrid_distillation_model(dummy_operating_conditions)
    print(f"Hybrid Predicted MoleFractionTX: {predicted_tx}")
    print(f"Hybrid Predicted MoleFractionHX: {predicted_hx}")


