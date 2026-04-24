import json
import numpy as np
import os
import src.config as config

def save_evaluation(footwork_scores, head_position_scores, swing_control_scores, balance_scores, follow_through_scores, reference_feedback=None):
    evaluation = {
        "Footwork": {
            "score": np.mean(footwork_scores) * 10 if footwork_scores else 0,
            "feedback": "Ensure your front foot points towards the cover region."
        },
        "Head Position": {
            "score": np.mean(head_position_scores) * 10 if head_position_scores else 0,
            "feedback": "Keep your head still and over your front knee for better balance."
        },
        "Swing Control": {
            "score": np.mean(swing_control_scores) * 10 if swing_control_scores else 0,
            "feedback": "A high elbow is key to a good cover drive. Ensure a full swing."
        },
        "Balance": {
            "score": np.mean(balance_scores) * 10 if balance_scores else 0,
            "feedback": "Maintain a stable base and good spine angle throughout the shot."
        },
        "Follow-through": {
            "score": np.mean(follow_through_scores) * 10 if follow_through_scores else 0,
            "feedback": "Complete your swing with a high and full follow-through."
        }
    }

    import src.config as config
    with open(os.path.join(config.OUTPUT_DIR, 'evaluation.json'), 'w') as f:
        json.dump(evaluation, f, indent=4) 

    