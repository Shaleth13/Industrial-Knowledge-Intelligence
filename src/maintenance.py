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
        status = []
        for score in predictions["Health_Score"]:
            if score >= 70:
                status.append("Healthy")
            elif score >= 40:
                status.append("Warning")
            else:
                status.append("Critical")
    
        predictions["Status"] = status
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

