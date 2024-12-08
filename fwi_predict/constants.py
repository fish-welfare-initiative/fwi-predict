# Project-wide constants

TIMEZONE = 'Asia/Kolkata'

WQ_RANGES = {
    'do_mg_per_L': {
        'required': {
            'morning': (3, 5),
            'evening': (8, 12)
        },
        'ideal': {
            'morning': (3, 5),
            'evening': (8, 10)
        }
    },
    'ph': {  # Can also do morning evening to have consistent interface
        'required': (6.5, 8.5),
        'ideal': (7, 8)
    },
    'ammonia_mg_per_L': {
        'required': (0, 0.5),
        'ideal': (0, 0.15)
    },
    'turbidity_cm': {
        'required': (20, 50),
        'ideal': (30, 40)
    }
}