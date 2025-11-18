from experiments.sa_frozen_lake import run_frozen_lake_experiment

def run_experiment():
    """Функция для запуска всех экспериментов на среде FrozenLake."""
    print("Starting FrozenLake experiments...")
    run_frozen_lake_experiment()
    print("Experiments completed!")

if __name__ == "__main__":
    run_experiment()
