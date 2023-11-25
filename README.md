# ASL-classifier

This is a project that aims to translate American Sign Language (ASL) alphabet in real-time. The backbone of the project is a convolutional neural network.

Results: the program easily recognises about 24/26 characters. It has issues with 'V' and 'X', perhaps due to their similarity to other characters
in the dataset.

Important files in the repository:
  - best_acc_model (the model that the program currently uses and that was picked based on accuracy on the test set)
  - ASLNet.py (CNN that is trained)
  - train.py (script that performs the training of the CNN)
  - hand_landmarks.py (the main script that runs the live recogniser. captures the hand automatically and displays the character and % of confidence)
  - sign_recognition.py (similar to hand_landmarks but uses a different model and has a static frame for the hand)
