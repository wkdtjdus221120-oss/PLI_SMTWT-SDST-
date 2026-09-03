from sa_smtwt_sdst import solve_sa
from run_common import common_parser, load_instance_or_exit, print_result


def main() -> None:
    parser = common_parser("Run Simulated Annealing on SMTWT-SDST.")
    parser.add_argument("--initial-temperature", type=float, default=None)
    parser.add_argument("--cooling-rate", type=float, default=0.95)
    parser.add_argument("--steps-per-temperature", type=int, default=None)
    parser.add_argument("--insertion-probability", type=float, default=0.50)
    args = parser.parse_args()

    instance = load_instance_or_exit(args.instance)
    result = solve_sa(
        instance,
        time_limit=args.time_limit,
        max_evaluations=args.max_evaluations,
        seed=args.seed,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        steps_per_temperature=args.steps_per_temperature,
        insertion_probability=args.insertion_probability,
    )
    print_result(result, details=args.details)


if __name__ == "__main__":
    main()
