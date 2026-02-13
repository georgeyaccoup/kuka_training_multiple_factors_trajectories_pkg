import numpy as np

class EnergyTeacher:
    """
    Implements the Linear Regression Model from Equation 12 of the paper.
    Acts as the 'Baseline' for the RL Agent to beat.
    """
    def __init__(self):
        # Coefficients from Source [cite: 211-213]
        self.intercept = 2084.402
        
        self.coef_payload = 34.609
        
        # Joint Positions (JP1 to JP6)
        self.coef_jp = [
            27.236,    # JP1
            576.884,   # JP2
            -360.930,  # JP3
            184.189,   # JP4
            -275.645,  # JP5
            -155.541   # JP6
        ]
        
        # Joint Velocities (JV1 to JV5 - JV6 not listed in eq snippet, assuming 0 or minor)
        self.coef_jv = [
            135.878,   # JV1
            -200.208,  # JV2
            -120.387,  # JV3
            267.384,   # JV4
            116.782,   # JV5
            0.0        # JV6
        ]

    def predict(self, payload, joints, velocities):
        """
        Calculates E_pred (Predicted Energy in Joules)
        """
        energy = self.intercept
        energy += self.coef_payload * payload
        
        # Add Joint Position effects
        for i in range(6):
            if i < len(joints):
                energy += self.coef_jp[i] * joints[i]
                
        # Add Joint Velocity effects (using magnitude/abs as per standard energy logic)
        for i in range(6):
            if i < len(velocities):
                energy += self.coef_jv[i] * abs(velocities[i])
                
        return energy