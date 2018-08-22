from timeit import default_timer as timer

# Class label for real edges
REAL = 1

# Class label for fake edges
FAKE = 0

# The name of the file containing the training data
TRAIN_FILE = "train.txt"

# The name of the file containing the test data
TEST_FILE = "test-public.txt"

# The prefix of the file that we save our predictions to
SAVE_FILE = "predictions-"

# The maximum number of files we can save
MAX_FILES = 100

# How maximum bound on how long each prediction can take
# (This is important for NeighbourClassifier)
TIME_LIMIT = 200.0

# The size of the batches passed to the neural network in TensorFlow
BATCH_SIZE = 100

# The number of epochs to run the neural network in TensorFlow for
EPOCHS = 100

# The limit for the number of training instances of each type (REAL + FAKE)
TRAINING_LIMIT = 25000

# The limit for the number of development instances of each type (REAL + FAKE)
DEV_LIMIT = 1000
