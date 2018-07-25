import joblib

# {1.: 'palm', 0.0: 'rock', 2.: 'fist', 3.: '1', 4.: '2'}

gestures_dict = {(1., 3., 4.): 'palm-1-2',
                             (2., 4., 0.): 'fist-2-rock',
                             (2., 3., 0.): 'fist-1-rock',
                             (1., 2., 0.): 'palm-fist-rock',
                             (1., 2., 1.): 'palm-fist-palm'}

joblib.dump(gestures_dict , 'gestures_combination_dict')

