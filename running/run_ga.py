from ga_smtwt_sdst import solve_ga
from run_common import common_parser, load_instance_or_exit, print_result


def main() -> None:
    parser = common_parser("Run Genetic Algorithm on SMTWT-SDST.")
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--crossover-rate", type=float, default=0.90)
    parser.add_argument("--mutation-rate", type=float, default=0.20)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=2)
    args = parser.parse_args()

    instance = load_instance_or_exit(args.instance)
    result = solve_ga(
        instance,
        time_limit=args.time_limit,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        population_size=args.population_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        tournament_size=args.tournament_size,
        elite_count=args.elite_count,
    )
    print_result(result, details=args.details)


if __name__ == "__main__":
    main()
