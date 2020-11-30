from src.performance import random_direct_performance, random_char_performance


def drive(num_samples):
    print("Direct:")

    print("Species")
    random_direct_performance(num_samples, "species")
    print("Edibility")
    random_direct_performance(num_samples, "edibility")

    print("Characteristics:")

    print("Species")
    random_char_performance(num_samples, "species")
    print("Edibility")
    random_char_performance(num_samples, "edibility")


if __name__ == "__main__":
    drive(100)
