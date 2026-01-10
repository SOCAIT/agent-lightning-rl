nutrition_schema = {
    "type": "object",
    "properties": {
        "dailyMealPlans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "minimum": 0, "maximum": 6},
                    "meals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "calories": {"type": "number"},
                                "proteins": {"type": "number"},
                                "carbs": {"type": "number"},
                                "fats": {"type": "number"},
                                "sequence": {"type": "integer"},
                            },
                            "required": ["name","calories","proteins","carbs","fats","sequence"]
                        }
                    }
                },
                "required": ["day","meals"]
            }
        }
    },
    "required": ["dailyMealPlans"]
}

meal_schema = {

    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "calories": {"type": "number"},
        "proteins": {"type": "number"},
        "carbs": {"type": "number"},
        "fats": {"type": "number"},
        "sequence": {"type": "integer"},
    },
    "required": ["name","calories","proteins","carbs","fats","sequence"]
}

daily_meal_plan_schema = {
    "type": "object",
    "properties": {
        # "day": {"type": "integer", "minimum": 0, "maximum": 6},
        "meals": {"type": "array", "items": meal_schema}
    },
    "required": ["meals"]
}


workout_one_week_schema = {
    "type": "object",
    "properties": {
        "workouts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "minimum": 0, "maximum": 6},
                    "workout_name": {"type": "string"},
                    "exercises": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "exercise": {"type": "string"},
                                "sets": {"type": "integer", "minimum": 1},
                                "reps": {"type": "integer", "minimum": 1},
                                "restTime": {"type": "string"}
                            },
                            "required": ["exercise","sets","reps","restTime"]
                        }
                    }
                },
                "required": ["day","workout_name","exercises"]
            }
        }
    },
    "required": ["workouts"]
}

