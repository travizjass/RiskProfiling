import os
import json
import csv

# Define file paths
data_folder = "withoutzip"
icd_mapping_file = "icd_mapping.json"
zip_data_file = "all_zip_data.json"
output_csv = "risk_profiling_data.csv"

# Load ICD mapping
with open(icd_mapping_file, "r") as file:
    icd_mapping = json.load(file)

# Load ZIP data
with open(zip_data_file, "r") as file:
    zip_data = {entry["zipCode"]: entry for entry in json.load(file) if "zipCode" in entry}

# Function to calculate BMI
def calculate_bmi(weight_lbs, height_inches):
    try:
        weight_kg = float(weight_lbs) * 0.453592
        height_m = float(height_inches) * 0.0254
        return round(weight_kg / (height_m ** 2), 2)
    except (ValueError, TypeError, ZeroDivisionError):
        return None

# Prepare CSV columns
columns = [
    "id", "patient_id", "age", "bmi", "bp_systolic", "bp_diastolic", "respiration", "temperature",
    "height", "weight", "pulse", "oxygen_saturation", "icd_code", "severity_score", "no_falls",
    "no_of_hospitalizations", "emergency_visits", "high_risk_drugs_count",
    "infections_count", "wounds_count", "zip_code", "averageIncome", "populationDensity",
    "educationPercentage", "employmentPercentage", "publicInsurancePercentage", "noInsurancePercentage"
]

# Open CSV file for writing
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()

    # Iterate through JSON files in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(data_folder, filename)
            with open(filepath, "r") as file:
                data = json.load(file)

                # Extract ICD code and severity score
                icd_codes = data.get("icd_codes", [])
                icd_code = icd_codes[0] if icd_codes else None
                severity_score = icd_mapping.get(icd_code, {}).get("severity")

                # Extract ZIP-related data
                zip_code = data.get("zip")
                zip_code = zip_code[:5] if zip_code else None
                zip_info = zip_data.get(zip_code, {})
                average_income = zip_info.get("averageIncome")
                population_density = zip_info.get("populationDensity")
                education_percentage = zip_info.get("educationPercentage")
                employment_percentage = zip_info.get("employmentDetails", {}).get("Employed", {}).get("percentage")
                public_insurance_percentage = zip_info.get("healthInsuranceDetails", {}).get("Public Insurance", {}).get("percentage")
                no_insurance_percentage = zip_info.get("healthInsuranceDetails", {}).get("No Insurance", {}).get("percentage")

                # Extract required fields
                record = {
                    "id": data.get("id"),
                    "patient_id": data.get("patient_id"),
                    "age": data.get("age"),
                    "bmi": calculate_bmi(
                        data.get("recent_visits", {}).get("weight_lbs"),
                        data.get("recent_visits", {}).get("height_inches")
                    ),
                    "bp_systolic": data.get("recent_visits", {}).get("bp_systolic"),
                    "bp_diastolic": data.get("recent_visits", {}).get("bp_diastolic"),
                    "respiration": data.get("recent_visits", {}).get("respiration"),
                    "temperature": data.get("recent_visits", {}).get("temperature"),

                    "height": data.get("recent_visits", {}).get("height_inches"),
                    "weight": data.get("recent_visits", {}).get("weight_lbs"),
                    "pulse": data.get("recent_visits", {}).get("pulse"),
                    "oxygen_saturation": data.get("oxygen_saturation"),
                    "icd_code": icd_code,
                    "severity_score": severity_score,
                    "no_falls": len(data.get("falls", [])),
                    "no_of_hospitalizations": data.get("no_of_hospitalization"),
                    "emergency_visits": data.get("emergency_visits"),
                    "high_risk_drugs_count": data.get("high_risk_drugs_count"),
                    "infections_count": data.get("infections_count"),
                    "wounds_count": data.get("wounds_count"),
                    # Use ZIP-related data
                    "zip_code": zip_code,
                    "averageIncome": average_income,
                    "populationDensity": population_density,
                    "educationPercentage": education_percentage,
                    "employmentPercentage": employment_percentage,
                    "publicInsurancePercentage": public_insurance_percentage,
                    "noInsurancePercentage": no_insurance_percentage
                }

                # Write record to CSV
                writer.writerow(record)

print(f"CSV file has been generated at {output_csv}")