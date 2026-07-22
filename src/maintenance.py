import pandas as pd

class MaintenanceAdvisor:

    def __init__(self):
        pass

    def health_score(self, predictions):
        max_rul = predictions["Predicted_RUL"].max()
        predictions["Health_Score"] = (predictions["Predicted_RUL"] / max_rul) * 100
        predictions["Health_Score"] = predictions["Health_Score"].round(1)
        return predictions

    def health_status(self, predictions):
        conditions = [
            predictions["Health_Score"] >= 70,
            predictions["Health_Score"].between(40, 70),
            predictions["Health_Score"] < 40]

        status = [
            "Healthy",
            "Warning",
            "Critical"]

        predictions["Status"] = pd.Series(pd.Categorical(pd.cut(predictions["Health_Score"], bins=[-1,40,70,100], labels=["Critical","Warning","Healthy"]))).astype(str)
        return predictions

    def recommendation(self, predictions):
        advice = []
        for score in predictions["Health_Score"]:
            if score >= 70:
                advice.append("No maintenance required.")
            elif score >= 40:
                advice.append("Schedule inspection within next few cycles.")
            else:
                advice.append("Immediate maintenance recommended.")

        predictions["Recommendation"] = advice
        return predictions

    def analyze(self, predictions):
        predictions = self.health_score(predictions)
        predictions = self.health_status(predictions)
        predictions = self.recommendation(predictions)
        return predictions

//USAGE
from maintenance import MaintenanceAdvisor

advisor = MaintenanceAdvisor()
final_result = advisor.analyze(latest)
print(final_result.head())
