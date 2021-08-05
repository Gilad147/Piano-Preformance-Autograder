import Performance_class
import numpy as np


def create_random_mistakes(path, name, n, max_noise, max_percentage, min_noise=0, min_percentage=0.5):
    flawed_performances = []
    for i in range(n):
        performance = Performance_class.Performance(path, name, "com", path)
        performance.mistakes_generator("rhythm", np.random.uniform(min_noise, max_noise), np.random.uniform(min_percentage, max_percentage))
        performance.mistakes_generator("duration", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage), False)
        performance.mistakes_generator("velocity", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage), False)
        performance.mistakes_generator("pitch", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage), False)
        flawed_performances.append(performance)

    return flawed_performances
