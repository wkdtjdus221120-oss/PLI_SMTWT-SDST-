from dde_smtwt_sdst import solve_dde
from run_common import common_parser, load_instance_or_exit, print_result


def main() -> None:
    parser = common_parser("Run Discrete Differential Evolution on SMTWT-SDST.")
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--differential-weight", type=float, default=0.60)
    parser.add_argument("--crossover-rate", type=float, default=0.90)
    args = parser.parse_args()

    instance = load_instance_or_exit(args.instance)
    result = solve_dde(
        instance,
        time_limit=args.time_limit,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        population_size=args.population_size,
        differential_weight=args.differential_weight,
        crossover_rate=args.crossover_rate,
    )
    print_result(result, details=args.details)


if __name__ == "__main__":
    main()
